import { distributions as dist, modeling as mdl } from 'tymc';
import * as samplers from 'tymc/samplers';
import { mmm } from 'tymc-marketing';
type Config = any;
type Dataset = any;

/** Match the guided PyMC model: max-scaled media/target, unscaled controls. */
export function buildTymcModel(data: Dataset, config: Config) {
  const columns = (names: string[], rows: number[][]) =>
    Object.fromEntries(
      names.map((name, c) => [name, rows.map((row) => row[c])]),
    );
  const raw = mmm.dataset({
    verbose: false,
    dates: data.dates,
    channels: columns(data.channels, data.x),
    controls: columns(data.controls, data.z),
    target: data.y,
  });
  const scaled = mmm.scale(raw, {
    channels: 'max',
    target: 'max',
    controls: 'none',
  });
  const spec = mmm.spec({
    media: {
      adstock: mmm.adstock.geometric({ l_max: config.lag, normalize: true }),
      saturation: mmm.saturation.logistic(),
    },
    ...(config.seasonality
      ? { seasonality: mmm.fourier.yearly({ order: 2 }) }
      : {}),
  });
  const priors = mmm.override_spec_priors(
    spec,
    mmm.pymc_marketing_spec_priors(spec, scaled),
    {
      'adstock.geometric.alpha': mmm.prior(
        dist.beta,
        config.adstockPrior ?? { alpha: 1, beta: 3 },
      ),
      'saturation.logistic.lam': mmm.prior(
        dist.gamma,
        config.saturationPrior ?? { alpha: 3, beta: 1 },
      ),
      'transformed_channel.coefficient': mmm.prior(dist.half_normal, {
        sigma: config.priorScale,
      }),
    },
  );
  return { spec, priors, scaled };
}

export function sampleTymcModel(
  data: Dataset,
  config: Config,
  onPhase: (phase: string) => void = () => {},
  onProgress?: (
    event: Parameters<NonNullable<mdl.SampleModelOptions['on_progress']>>[0],
    alpha?: number,
  ) => boolean | undefined,
  onPriorReport: (report: string) => void = () => {},
) {
  const setup = buildTymcModel(data, config);
  onPhase('Checking the model priors');
  const checked = mmm.check_priors(setup.spec, setup.priors, setup.scaled, {
    verbose: false,
  });
  onPriorReport(mmm.format_prior_check(checked));
  const built = mmm.build(setup.spec, setup.priors, setup.scaled);
  onPhase('Compiling TyMC model');
  const compileStart = performance.now();
  const compiled = mdl.compile_model(built.model);
  const alphaTarget = mdl
    .draw_targets(built.model)
    .find((target) => target.name === 'adstock.geometric.alpha')!;
  const compileSeconds = (performance.now() - compileStart) / 1000;
  onPhase('Sampling with TyMC');
  const sampleStart = performance.now();
  const chains = samplers.sample(compiled.logp_and_grad, {
    draws: config.draws,
    tune: config.tune,
    chains: config.chains,
    target_accept: config.targetAccept,
    seed: config.seed,
    initial_point: mdl.initial_point(built.model).data,
    on_progress: onProgress
      ? (event) =>
          onProgress(
            event,
            event.phase === 'draw'
              ? mdl.constrain_draw(built.model, event.position, [
                  alphaTarget,
                ])[0].data[0]
              : undefined,
          )
      : undefined,
    progress_every: 1,
    progress_interval_ms: 0,
  });
  const samplingSeconds = (performance.now() - sampleStart) / 1000;
  onPhase('Computing posterior predictions and diagnostics');
  const result = mdl.assemble_result(built.model, chains);
  return {
    fitted: { ...built, prior_check: checked, result },
    compileSeconds,
    samplingSeconds,
    priorReport: mmm.format_prior_check(checked),
  };
}
