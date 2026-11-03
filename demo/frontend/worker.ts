import { mmm } from 'tymc-marketing';
import { sampleTymcModel } from './tymc-model';

self.onmessage = ({data}) => {
  try {
    const {world} = data;
    const n = world.context_weeks;
    const date = (i: number) => new Date(Date.UTC(2020, 0, 6 + 7*i)).toISOString().slice(0,10);
    const dates = world.outcome.map((_: number,i: number) => date(i));
    const dataset = {dates: dates.slice(0,n), channels: world.channels, controls: ['promotion'], x: world.spend.slice(0,n), z: world.controls.slice(0,n), y: world.outcome.slice(0,n)};
    const config = {lag: 8, seasonality: true, priorScale: 2, chains: 4, tune: 1000, draws: 1000, targetAccept: 0.99, seed: 42};
    const run = sampleTymcModel(dataset, config,
      phase => self.postMessage({type:'phase', phase}),
      event => { if (event.iteration % 25 === 0) self.postMessage({type:'progress', chain:event.chain+1, phase:event.phase, iteration:event.iteration+1}); return true; });
    self.postMessage({type:'phase', phase:'Forecasting the held-out weeks'});
    const columns = Object.fromEntries(world.channels.map((name: string, j: number) => [name, world.spend.slice(n).map((r: number[]) => r[j])]));
    const future = mmm.dataset({verbose:false, dates:dates.slice(n), channels:columns, controls:{promotion:world.controls.slice(n).map((r: number[])=>r[0])}});
    const scaled = mmm.scale_like(future, run.fitted.scaled_data.scales);
    const prediction = mmm.unscale(mmm.predict(run.fitted, scaled, {include_last_observations:true,seed:142}).expected_target,run.fitted.scaled_data.scales);
    const width = world.outcome.length-n;
    const count = prediction.values.length/width;
    const mean = Array.from({length:width},(_,t) => {let s=0; for(let d=0;d<count;d++) s+=prediction.values[d*width+t]; return s/count;});
    const summary = run.fitted.result.summary;
    const rhats = summary.map((v:any)=>v.r_hat).filter(Number.isFinite);
    const divergences = run.fitted.result.chains.reduce((sum:number,c:any)=>sum+Array.from(c.divergent).filter(Boolean).length,0);
    self.postMessage({type:'result',mean, seconds:run.compileSeconds+run.samplingSeconds, divergences, maxRhat:rhats.length?Math.max(...rhats):null, draws:count});
  } catch (err) {self.postMessage({type:'error',message:err instanceof Error?err.message:String(err)});}
};
