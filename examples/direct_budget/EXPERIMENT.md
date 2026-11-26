# Direct allocation: synthetic experiment

This is a research result, not evidence of client performance. It uses the
synthetic prior in `pymc_mfm.budget.synthetic`. The network returns allocations
only. The simulator computes outcome during training and evaluation.

## Run

- 40,000 training worlds, seed 110; 2,000 validation worlds, seed 111.
- Four channels, 64 historical periods; one-period future decision.
- 30 epochs; three imitation warmup epochs, followed by decision-outcome loss.
- The validation set selected the checkpoint. Test results did not select it.
- In each training and test set, constraints include no limits, one channel cap,
  and a feasible box around a random allocation. The target spends all budget.
- Decision loss uses each simulated world's true response mechanism. Those
  privileged parameters are not available as network inputs.

The saved local checkpoint was re-tagged `direct-budget-v2` after the run to
record its actual last-spend input semantics. Tensor values did not change.
Weights are not tracked in Git. Run the training script to produce new weights.

## Held-out results

On 4,000 new worlds from the same prior, seed 112, average regret against the
world-specific global optimum was 0.908 for the network and 4.213 for feasible
projected equal allocation. Regret is measured in synthetic outcome units; the
oracle has access to the world's hidden parameters. The network had lower
regret in 87.98% of paired worlds. The paired mean difference was 3.305 with
standard error 0.062. This comparison is specific to this simulator and these
baselines.

For unconstrained worlds, network and equal-allocation mean regret were 1.410
and 5.883 (972 worlds). For a single channel cap, they were 1.117 and 4.438
(1,006 worlds). For boxed constraints, they were 0.562 and 3.298 (2,022 worlds).
The largest observed total-budget error was 0.000016 in budget units. It comes
from float32 rounding, so downstream systems must use an appropriate tolerance.

A separate 4,000-world stress set, seed 113, adds unobserved demand that affects
historical spend and outcome. Network mean regret rose to 2.429, compared with
4.281 for equal allocation. This is one synthetic stress, not proof of robustness
to confounding. It does not establish causal identification from observed data.

Additional 2,000-world-per-setting exploratory tests with 2, 3, 6 and 8 channels
and 52 or 104 periods used the same family of simulated mechanisms. The model
had lower mean regret than the equal-allocation baseline in each setting. These
are out-of-training-shape tests, not evidence for untested mechanisms or real
marketing systems.

The prior omits several material mechanisms, including interactions, time-varying
response parameters, observed controls and hidden demand. The outcomes do not
validate calibration, uncertainty estimates, real-world generalization or
business impact. The model has no built-in check that input data match its prior.

## Controlled model, synthetic prior

The controlled version uses 40,000 synthetic worlds, seed 220, and a separate
2,000-world validation set, seed 221. Thirty training epochs select epoch 23 by
validation mean regret. Each of four channels is independently positive (60%),
null (20%) or harmful (20%). Two observed controls affect historical spend and
outcome. Channel responses are additive and concave; harmful effects are linear.
A constrained optimizer computes the synthetic optimum, including tied null
channels. This is a limited prior and does not cover arbitrary response shapes,
interactions or structural shifts.

On 4,000 held-out worlds, seed 222, mean regret was 0.768 versus 10.248 for the
feasible equal baseline. On 4,000 worlds with an additional unobserved shock,
seed 223, it was 2.448 versus 10.525. These synthetic units and priors differ
from the earlier experiment; do not compare the numeric regrets across priors.
The checkpoint was trained with float32 projection; the saved weights were
re-evaluated after the CPU projection changed to float64. The maximum observed
absolute budget-sum error was 0.000016, with maximum bound error 0.000004.

## Joint-association model, expanded prior

A third checkpoint adds regularized joint-association features and positive
linear channels. It uses 40,000 training worlds (seed 420), 2,000 validation
worlds (seed 421), and 30 epochs. The validation-selected checkpoint is epoch
30. These features are internal network inputs, not calibrated causal effects.

On 4,000 held-out same-prior worlds (seed 422), mean regret was 0.741 for the
network, 2.881 for an OLS-based linear allocation policy with the same observed
controls and limits, and 10.821 for feasible equal allocation. The network had
lower regret than OLS in 63.15% of worlds and higher regret in 35.35%. The
remaining cases are numerical ties. This is a same-prior simulation, not a
client benchmark. Reproduce the comparison with:

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_controlled.py \
  --model artifacts/direct-budget/joint.safetensors --seed 422 --positive-linear
```

The positive-linear option uses positive concave, null, harmful linear and
positive linear channels. The synthetic reference solver's optimality
conditions were checked across 10,000 additional generated worlds. The model
still uses an additive single-period prior and does not identify hidden
confounding from historical observations.

## Post-development linear stress

The self-contained `evaluate_ood.py` adds a different linear simulator with
correlated media spend, three observed controls, channel-specific carryover,
positive, null and negative effects, and optional unobserved demand. This stress
was designed after examining earlier experiments; it is not a blinded mechanism
holdout. All methods receive the same historical observations and controls.
Each method allocates the same fixed total with no channel-specific limits.

For 2,048 worlds with seed 880321 and no hidden demand, mean regret was 0.711
for the joint network, 0.342 for OLS and 8.755 for equal allocation. For a
separate 2,048-world draw with seed 880322 and hidden-demand strength 0.7,
mean regret was 1.464, 0.525 and 9.186. The two draws use different seeds;
their difference does not isolate the effect of hidden demand. OLS performed
better than the network in this stress. These outcome units are specific to
this simulator and cannot be compared with those of other tests.

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_ood.py \
  --model artifacts/direct-budget/joint.safetensors --seed 880321
PYTHONPATH=src python examples/direct_budget/evaluate_ood.py \
  --model artifacts/direct-budget/joint.safetensors --seed 880322 --hidden-strength 0.7
```

## Persistent controls and mixed geometries

A structured-prior checkpoint uses 40,000 four-channel worlds with correlated
media and persistent controls, seed 520. A separate mixed-geometry checkpoint
uses 10,000 worlds for each of four channel/history/control geometries, seed
12620 with disjoint geometry-specific seeds. Both use positive linear channels
and joint-association features. Validation selects checkpoints before the
reported test draws. These are simulation experiments, not client validation.

On 4,096 new worlds per condition from the post-development linear stress,
mean regret for structured-four/mixed/OLS was 0.457/0.498/0.258 without hidden
demand, and 0.972/1.204/0.521 with hidden demand in a separate draw. The paired
mixed-minus-structured difference was 0.041 (SE 0.009) and 0.232 (SE 0.018),
respectively. The simulator family informed development, and the standard
errors do not account for model selection. Different seeds are used across
the two conditions; differences across conditions are not paired effects.

On 1,000 same-prior worlds per geometry, mean regret for structured-four/mixed
was 0.247/0.214 with 2 channels, 0.564/0.560 with 3, 0.937/0.976 with 4 and
1.354/1.384 with 6. The mixed model extends training coverage but does not
establish superior decisions. It is not selected as a general replacement.

For example, the post-development test is reproducible with a locally trained
checkpoint:

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_ood.py \
  --model artifacts/direct-budget/mixed.safetensors --seed 1180321 --worlds 4096
```

## Four-period direct planning

A separate model outputs four budget vectors directly. It was initialized from
the structured one-period checkpoint, then trained on 16,000 structured
synthetic worlds (seed 24680) to maximize the sum of outcomes over four
periods. Each period has a fixed total and channel bounds. Carryover links
periods. The validation-selected checkpoint was epoch 12 of 15. No forecasts
or contributions are returned by inference.

The evaluator uses projected gradient to find a feasible synthetic reference.
For this additive concave simulator, the Frank-Wolfe gap gives an upper bound
on the remaining optimization error. It reports an interval for regret rather
than claiming that the numerical search always finds the exact optimum.

On 2,048 fresh same-prior worlds, seed 54680, mean regret bounds were 5.806–5.861
for the planning model, 6.797–6.852 for the repeated one-period network and
25.446–25.501 for a static OLS rule. The paired planning gain over the
one-period rule was 0.991 (SE 0.130); planning won 39.9% and lost 53.7% of
worlds. The rest were ties. The mean optimization certificate gap was 0.055;
the maximum was 2.179.

On a separate 2,048-world draw with hidden demand, seed 54681, the planning
regret bounds were 9.730–9.798 versus 9.062–9.131 for the one-period network.
Planning's mean gain was -0.667 (SE 0.135); it won 33.9% and lost 62.6% of
worlds. These tests followed a smaller probe and use different seeds across
conditions. Their standard errors do not account for research selection. The
results do not establish real-world performance, robustness to hidden demand,
or optimality under other mechanisms.

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_planning_ols.py \
  --model artifacts/direct-budget/planning.safetensors \
  --baseline artifacts/direct-budget/structured.safetensors --seed 54680
```

## Hidden-demand mixture and linear planning stress

A second four-period network was trained on 16,000 worlds mixing hidden-demand
strengths 0, 0.25, 0.5 and 0.75. The hidden factor is not supplied to the
network. On matched draws of 2,048 worlds each, the mixture-trained network's
mean outcome gain over the original planning network was -0.488 (SE 0.101)
without hidden demand, +1.058 (SE 0.143) at strength 0.5 and +1.804 (SE 0.157)
at strength 0.75. Different strengths use different seeds. These simulations
share the training generator family; the result does not establish robustness
to arbitrary unobserved confounding.

A separate post-development linear carryover stress has two channels, three
observed controls and four budget periods. The synthetic linear optimum is
computed analytically. On 2,048 worlds without hidden demand, mean regret for
original planning / mixture-trained planning / repeated one-period / OLS was
4.244 / 4.262 / 3.665 / 2.302. On a separate 2,048-world draw with hidden
strength 0.7, it was 7.052 / 6.540 / 5.722 / 3.119. The planning networks
lagged both baselines in these tests. The simulator family resembles earlier
exploratory tests, so it is not a blinded unseen mechanism. Units are synthetic
and not comparable to the preceding concave-prior results.

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_planning_ood.py \
  --model artifacts/direct-budget/planning-robust.safetensors \
  --baseline artifacts/direct-budget/structured.safetensors --seed 88990
```

## Complete future schedule

The first planning architecture used the current period's total and limits,
the mean total and the period index. It could not distinguish two schedules
that share those values but differ in later periods. The schedule architecture
adds a bidirectional per-channel encoder over all future totals and bounds.
Training uses 16,000 worlds with period-varying limits, seed 98760, and mixes
hidden-demand strengths. The validation-selected checkpoint is epoch 14 of 15.
It returns budgets only and is trained on four periods.

On 2,048 matched varying-bound worlds without hidden demand, the mean outcome
gain of schedule over the previous mixture-trained plan was -0.156 (SE 0.040).
With hidden-demand strength 0.5 on a separate draw, it was +0.166 (SE 0.044).
The conditions use different seeds. On 2,048 worlds per condition from the
post-development linear carryover stress, the corresponding gains were -0.149
(SE 0.050) without and -0.129 (SE 0.055) with hidden demand. OLS also had
lower mean regret than both planning networks in that stress. In the hidden
stress, the schedule network won 68.0% and lost 22.9% of worlds under a
numerical tie threshold, despite the negative mean gain. The SEs do not account
for research selection. These synthetic results do not establish a general
advantage, real-world performance or identification of hidden demand.

The matched varying-bound comparison can be reproduced with:

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_schedule.py \
  --schedule artifacts/direct-budget/schedule.safetensors \
  --previous artifacts/direct-budget/planning-robust.safetensors \
  --baseline artifacts/direct-budget/structured.safetensors --seed 198760
```

Use `evaluate_schedule_ood.py` with the same model arguments and `--seed 198763
--hidden 0.7` to reproduce the linear hidden-demand stress.

## Broader one-period prior

A mixture of the structured and broader synthetic generators was used to train
40,000 worlds with 2, 3, 4 and 6 channels, seed 22620, 30 epochs. The selected
checkpoint was epoch 23. The broader generator varies spend scale, channel
correlation, temporal dependence, controls, hidden demand, noise and channel
response type. These are still assumptions about simulated worlds.

On 4,096 matched worlds per condition, the diverse model's mean outcome gain
versus the earlier structured model was +0.0665 (SE 0.0180) on the known
linear stress without hidden demand and +0.2308 (SE 0.0293) with hidden demand.
However, mean regret for diverse / structured / constrained OLS was
0.4254 / 0.4919 / 0.2888 and 0.8507 / 1.0815 / 0.5843 respectively, so OLS
remained better on both. On the structured simulator, the diverse model's
paired gain was -0.0505 (SE 0.0095) without hidden demand and +0.0600 (SE
0.0114) with it. On the broader generator those gains were +0.4961 (SE 0.0782)
and +0.5560 (SE 0.0527). Conditions use different seeds and outcome scales;
they cannot identify the effect of changing hidden demand. The linear stress
family informed development, the SEs exclude research selection, and no test
establishes reliable real-world decisions or identifies unobserved confounding.

```sh
PYTHONPATH=src python examples/direct_budget/train_diverse.py \
  --output artifacts/direct-budget/diverse.safetensors \
  --worlds-per-geometry 10000 --epochs 30 --seed 22620
PYTHONPATH=src python examples/direct_budget/evaluate_diverse.py \
  --candidate artifacts/direct-budget/diverse.safetensors \
  --reference artifacts/direct-budget/structured.safetensors \
  --prior linear_ood --seed 2190321 --worlds 4096
```

For `linear_ood`, the evaluator uses its fixed two-channel, three-control
geometry regardless of the geometry options used for the other priors.

## Knowing limits before versus after the decision

We compared the diverse policy with the same policy first run without limits,
then Euclidean-projected onto the feasible set. The latter is one concrete
post-processing method, not all possible downstream optimizers. Both use the
same historical data and must spend the entire total. In every four-channel
world, the reference allocation is [19%, 30%, 30%, 21%] of the total, the first
channel is capped at 10%, and each channel may change by at most 10 percentage
points of the total. The exact constrained oracle is recomputed for each world.

On 4,096 matched broader-prior worlds without hidden demand, conditioned /
post-projected / OLS mean regret was 0.367 / 2.301 / 0.383. The conditioned
policy's paired outcome gain over post-projection was 1.934 (SE 0.051). On a
separate 4,096-world draw with hidden demand strength 0.5, corresponding regret
was 0.407 / 2.178 / 0.476 and paired gain was 1.771 (SE 0.050). These are
synthetic outcome units, not comparable across draws as a confounding effect.
Maximum budget error across either method was 0.000023 spend units and maximum
bound violation 0.000006, reflecting float precision. These selected constraints
and synthetic family cannot establish that conditioning always wins or that a
separate constrained optimizer would perform worse. Neither method establishes
real-world causal identification.

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_posthoc.py \
  --model artifacts/direct-budget/diverse.safetensors \
  --seed 36001 --worlds 4096
```

### Linear out-of-prior stress

The same comparison on the previously studied two-channel linear carryover
stress produced the opposite ranking. A 20% / 80% reference, first-channel
cap of 10% and maximum absolute change of 20% of total make the first channel's
feasible interval 0% to 10%. On 4,096 worlds per condition, conditioned /
post-projected / OLS mean regret was 0.0684 / 0.0571 / 0.0322 without hidden
demand, and 0.1706 / 0.1416 / 0.0580 with hidden demand on a separate seed.
Paired conditioned-minus-post-projected outcome differences were -0.0113
(SE 0.0019) and -0.0291 (SE 0.0032), respectively. Maximum budget error was
0.000011 spend units, and bound violation 0.000004. This test uses different
geometry and limits from the broader-prior comparison, and its mechanism was
examined during development. It does not establish a general ranking or test
an optimal posthoc optimizer. Outcomes and hidden-demand settings use different
seeds and must not be interpreted as a causal comparison of hidden demand.

```sh
PYTHONPATH=src python examples/direct_budget/evaluate_posthoc_ood.py \
  --model artifacts/direct-budget/diverse.safetensors --seed 36101 --worlds 4096
```

## Mixed linear and nonlinear training

A second alternative adds synthetic linear, signed or null responses with
variable carryover, media correlation, persistence, noise and hidden demand.
Training used 3,000 worlds per generator and geometry, 30,000 worlds total,
five geometries, seed 27182 and 20 epochs; validation selected epoch 20. The
linear training generator is distinct from the evaluation stress but that
stress family informed development.

On 4,096 matched worlds per condition of the known linear stress, mean regret
for mixed / diverse / OLS was 0.3768 / 0.4272 / 0.2853 without hidden demand
and 0.6676 / 0.7637 / 0.5224 with hidden demand on a distinct seed. Paired
mixed-minus-diverse gains were +0.0504 (SE 0.0071) and +0.0961 (SE 0.0115).
On the structured prior the same gains were -0.2693 (SE 0.0223) and +0.0194
(SE 0.0210); on the broader prior -0.1504 (SE 0.0432) and -0.0951 (SE 0.0346).
These last pairs likewise use separate seeds, with hidden strength 0 and 0.5.
The simulator families have different scales. SEs do not account for research
selection, and the results are not real-world validation. The mixed checkpoint
is an alternative, not a uniformly better model.

```sh
PYTHONPATH=src python examples/direct_budget/train_linear_mix.py \
  --output artifacts/direct-budget/linear-mix.safetensors \
  --worlds-per-geometry 3000 --epochs 20 --seed 27182 --threads 4
PYTHONPATH=src python examples/direct_budget/evaluate_diverse.py \
  --candidate artifacts/direct-budget/linear-mix.safetensors \
  --reference artifacts/direct-budget/diverse.safetensors \
  --prior linear_ood --seed 37101 --worlds 4096
```

## Order-aware history encoder

The earlier observation encoder pools per-period features, and its joint
regression and other summaries are permutation-invariant except for the
final-period spend input. The temporal alternative adds one- and two-period
lag features and their products with outcome before pooling. This supplies
some information about ordering without returning forecasts or contributions;
it is not a general sequence model or proof of identified carryover.

The temporal checkpoint was trained on 20,000 structured/diverse worlds across
four geometries, 20 epochs, seed 28282, with epoch 20 selected on validation.
The reference diverse checkpoint used 40,000 worlds and 30 epochs, so the
following comparison does not isolate architecture from training budget. On
4,096 matched known linear stress worlds per condition, mean regret for
temporal / diverse / OLS was 0.5303 / 0.4492 / 0.3225 without hidden demand
and 1.0714 / 0.8451 / 0.6179 on a separate hidden-demand draw. On structured
worlds, paired temporal-minus-diverse outcome gains were -0.0012 (SE 0.0099)
and -0.1796 (SE 0.0140); on diverse worlds, -0.0295 (SE 0.0209) and -0.0461
(SE 0.0260). Conditions use different seeds and scales. The known linear
family informed development, and SEs exclude research selection.

In a post-training sensitivity check on 1,000 synthetic histories, reversing
the order of all but the last observation changed the temporal model's budget
by mean L1 3.043 spend units; 40.8% changed by more than 0.0001. The earlier
model's mean L1 change was 0.000030, including numerical differences. This
perturbation can create unrealistic histories and demonstrates sensitivity,
not correct use of time, causal identification, or real-world improvement.

```sh
PYTHONPATH=src python examples/direct_budget/train_temporal.py \
  --output artifacts/direct-budget/temporal.safetensors \
  --worlds-per-geometry 5000 --epochs 20 --seed 28282 --threads 4
PYTHONPATH=src python examples/direct_budget/evaluate_temporal.py \
  --candidate artifacts/direct-budget/temporal.safetensors \
  --reference artifacts/direct-budget/diverse.safetensors \
  --prior linear_ood --seed 38101 --worlds 4096
```

### Same training worlds and budget

A subsequent comparison used the same generator code, 40,000 training worlds,
training data seed 22620, 30 epochs and checkpoint selection procedure for both
architectures. The temporal model selected epoch 30, the diverse reference epoch
23. Initializations and model sizes differ; this is one training seed, not an
estimate of variation across training runs. The temporal checkpoint's mean
validation regret was 1.1949 versus 1.2490 for the reference.

On 4,096 paired test worlds per condition, temporal-minus-reference mean outcome
gains were +0.0119 (SE 0.0041) and +0.0180 (SE 0.0073) in the known linear
stress without hidden demand and with strength 0.7, respectively. OLS still had
lower mean regret: temporal / reference / OLS were 0.3809 / 0.3928 / 0.2746
and 0.7825 / 0.8005 / 0.5873. On the structured prior, gains were +0.0216
(SE 0.0086) and -0.1170 (SE 0.0116); on the diverse prior, +0.0475 (SE 0.0223)
and +0.0573 (SE 0.0234). Those condition pairs use different seeds and the
latter hidden strength is 0.5. Simulator families have different scales.
The stress family informed development, and SEs quantify world sampling for
these particular fitted weights, not model selection or variation between
training runs. This comparison does not establish general or real-world benefit.

```sh
PYTHONPATH=src python examples/direct_budget/train_temporal.py \
  --output artifacts/direct-budget/temporal-matched.safetensors \
  --worlds-per-geometry 10000 --epochs 30 --seed 22620 --threads 4
PYTHONPATH=src python examples/direct_budget/evaluate_temporal.py \
  --candidate artifacts/direct-budget/temporal-matched.safetensors \
  --reference artifacts/direct-budget/diverse.safetensors \
  --prior linear_ood --seed 39101 --worlds 4096
```
