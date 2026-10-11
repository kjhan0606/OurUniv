"""Exact row partition of conditional raw factors, with full source support.

This bounds workspace without dropping source components or renormalizing a
spatial subvolume. Each observational row is complete before its conditional
log density is formed. Counts and proper priors are NOT repeated per batch.
"""
import copy
import time
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_volume_target import FreshRawSupport,raw_field_logpdf


class RowStreamedRawReadout:
    def __init__(self,source,population,observation,geometry,*,source_spacing,
                 volume_order,rows_per_batch=64):
        self.n=len(population)
        if self.n<1 or rows_per_batch<1 or any(len(x)!=self.n for x in observation.values()):
            raise ValueError('aligned nonempty observation rows and positive batch required')
        self.source=source;self.population=np.asarray(population);self.observation=observation
        self.rows_per_batch=rows_per_batch
        # One immutable spatial tree is shared across row views; never rebuild
        # a16-million-source tree for each batch or freeze state-local support.
        self.builder=FreshRawSupport(source['positions'],source['angular'],population,
            observation,geometry,source_spacing=source_spacing,volume_order=volume_order)
        self.value=jax.jit(lambda density,velocity,t,p,packs,o,src:raw_field_logpdf(
            density,velocity,t,p,packs,src,o,geometry,source_spacing=source_spacing,
            volume_order=volume_order,cut_order=256,cut_integration_axis=1,
            cut_marginal_tolerance=1e-12))

    def evaluate(self,density,velocity,tracer,population_white,progress=None):
        tic=time.monotonic();values=[];info=[]
        for start in range(0,self.n,self.rows_per_batch):
            stop=min(start+self.rows_per_batch,self.n)
            observation={k:v[start:stop] for k,v in self.observation.items()}
            builder=copy.copy(self.builder)
            builder.o=observation;builder.population=self.population[start:stop]
            packs,metadata=builder.build(velocity,tracer)
            packs=tuple({k:jnp.pad(v,(0,(-len(v))%65536),constant_values=False if k=='mask' else 0)
                         for k,v in pack.items()} for pack in packs)
            arguments=(density,velocity,tracer,population_white,packs,observation,self.source)
            compiled=self.value.lower(*arguments).compile()
            memory=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
            peak=stats.get('bytes_in_use',0)+memory.temp_size_in_bytes+memory.output_size_in_bytes
            if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
                raise MemoryError('row-streamed raw value lacks20percent device margin')
            result=np.asarray(compiled(*arguments))
            if result.shape!=(stop-start,) or not np.isfinite(result).all():
                raise FloatingPointError('every full row must have a finite conditional factor')
            values.append(result);info.append(dict(start=start,stop=stop,**metadata))
            if progress is not None:progress(info[-1])
        return np.concatenate(values),dict(seconds=time.monotonic()-tic,batches=info,
            components=sum(row['components'] for row in info),
            maximum_batch_components=max(row['components'] for row in info))
