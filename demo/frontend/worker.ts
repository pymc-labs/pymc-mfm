import { mmm } from 'tymc-marketing';
import { sampleTymcModel } from './tymc-model';

self.onmessage = ({data}) => {
  try {
    const {world} = data;
    const n = world.context_weeks;
    const date = (i: number) => new Date(Date.UTC(2020, 0, 6 + 7*i)).toISOString().slice(0,10);
    const dates = world.spend.map((_: number,i: number) => date(i));
    const dataset = {dates: dates.slice(0,n), channels: world.channels, controls: ['promotion'], x: world.spend.slice(0,n), z: world.controls.slice(0,n), y: world.outcome.slice(0,n)};
    const config = {lag: 8, seasonality: true, priorScale: 2, chains: 4, tune: 1000, draws: 1000, targetAccept: 0.99, seed: 42};
    const run = sampleTymcModel(dataset, config,
      phase => self.postMessage({type:'phase', phase}),
      event => { if (event.iteration % 25 === 0) self.postMessage({type:'progress', chain:event.chain+1, phase:event.phase, iteration:event.iteration+1}); return true; });
    self.postMessage({type:'phase', phase:'Estimating channel contributions'});
    const channelDraws = mmm.contributions_in_target_units(run.fitted).channels;
    const width = n, channels=world.channels.length;
    const count = channelDraws.values.length/(width*channels);
    if (!Number.isInteger(count) || count < 2 || channelDraws.shape[0] !== width || channelDraws.shape[1] !== channels) throw Error('Unexpected channel contribution dimensions');
    const mean:number[]=[], low:number[]=[], high:number[]=[];
    let score=0;
    for (let c=0;c<channels;c++) {
      const xs = Array.from({length:count}, (_,d) => {let total=0;for(let t=0;t<width;t++) total+=channelDraws.values[(d*width+t)*channels+c];return total/width;}).sort((a,b)=>a-b);
      let total=0, distance=0, pairs=0;
      for(let d=0;d<count;d++){total+=xs[d];distance+=Math.abs(xs[d]-world.contribution_truth[c]);pairs+=(2*d-count+1)*xs[d];}
      const quantile=(q:number)=>{const p=(count-1)*q,i=Math.floor(p);return xs[i]+(xs[Math.ceil(p)]-xs[i])*(p-i);};
      mean.push(total/count);low.push(quantile(.1));high.push(quantile(.9));
      score+=distance/count-pairs/(count*(count-1));
    }
    const contributionCrps=score/channels;
    const summary = run.fitted.result.summary;
    const rhats = summary.map((v:any)=>v.r_hat).filter(Number.isFinite);
    const divergences = run.fitted.result.chains.reduce((sum:number,c:any)=>sum+Array.from(c.divergent).filter(Boolean).length,0);
    self.postMessage({type:'result',mean, low, high, contributionCrps, seconds:run.compileSeconds+run.samplingSeconds, divergences, maxRhat:rhats.length?Math.max(...rhats):null, draws:count});
  } catch (err) {self.postMessage({type:'error',message:err instanceof Error?err.message:String(err)});}
};
