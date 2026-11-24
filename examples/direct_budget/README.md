# Direct budget allocation experiment

This experiment trains a separate network to output a budget allocation. It
is not a River checkpoint. The inputs are historical spend, historical outcome,
a total future budget, and lower and upper limits for each channel. The output
is an allocation that sums to the budget and satisfies the limits. It does not
return a forecast, a response curve, or a channel contribution.

The experiment samples synthetic worlds. In each world, random independent
historical spend produces noisy aggregate outcomes through additive concave
responses, channel-specific carryover, and a seasonal baseline. An analytical
constrained optimizer maximizes the next-period outcome in each world and
produces an oracle allocation. After a short imitation warmup, default training
optimizes the simulated outcome of the network allocation directly. The
optimizer and response parameters are not inputs to the network. An
imitation-only baseline is available with `--objective imitation`.
The future decision is a single period. The synthetic prior
has no interactions, changing response parameters, or hidden confounding. The
decision spends the full specified budget and does not account for profit,
channel costs beyond spend, or risk preferences. It does not establish
real-world causal identification or performance on actual marketing data.

Run with the `local` extra installed:

```sh
PYTHONPATH=src python examples/direct_budget/train.py \
  --output artifacts/direct-budget/decision.safetensors
```

The script selects a checkpoint using validation-world regret. It then evaluates
separate, newly generated test worlds against the synthetic global optimum,
feasible equal allocation and feasible recent-spend allocation. It also tests
an out-of-prior hidden-demand scenario. Results are written as JSON beside the
weights. Model weights are excluded from Git. Evaluate other priors, decision
horizons and real experiments before using any allocation in practice.

Example inference after training:

```python
import torch
from pymc_mfm.budget.io import load_direct_budget_model

model = load_direct_budget_model("artifacts/direct-budget/decision.safetensors")
# spend: [batch, weeks, channels]; outcome: [batch, weeks]
# budget: [batch]; lower and upper: [batch, channels], all in spend units
with torch.no_grad():
    allocation = model(spend, outcome, budget, lower, upper)
```

The model is trained with four channels and 64 weeks. Although the network
accepts other shapes, performance outside that geometry has not been established.
Limits must be jointly feasible: sum(lower) <= budget <= sum(upper).

For a JSON input, run `python examples/direct_budget/allocate.py --model
artifacts/direct-budget/decision.safetensors --input input.json`. The input
object has `channels` (unique names), `spend` (one row per week, in channel
order), `outcome` (one value per week), and `budget` (the next-period total).
Optional `lower` and `upper` arrays give limits in the same order and units.
Without limits, the lower bounds are zero and the upper bounds equal the total
budget. For example, setting the TV upper bound to 10 for a total of 100 limits
TV to 10%. The output contains the total and the allocation by channel only.

## Controlled-budget research version

`train_controlled.py` trains a separate architecture on worlds with observed
controls and a mixture of positive, null and harmful channel effects. The
`--positive-linear` option adds positive linear channel effects. The model
returns budgets directly. It residualizes spend and outcome against supplied
controls as an input feature. The optional `--joint-features` adds regularized
joint linear-association features; these are inputs to the decision network,
not outputs or calibrated causal effects. This does not establish that hidden
confounding has been removed. The model uses a one-period outcome objective and
a prior with additive channel effects. The supplied controls are historical,
not future scenarios.

```sh
PYTHONPATH=src python examples/direct_budget/train_controlled.py \
  --output artifacts/direct-budget/controlled.safetensors --positive-linear --joint-features
```

For this model, `allocate.py` also requires a `controls` array with one row
per historical period and one column per observed control. To explicitly use
no controls, provide one empty array per period. The same CLI returns only
channel budgets. The model is trained with two controls; zero controls or a
different number can work technically but are outside that training geometry.
The output spends the full specified budget, even when a channel has a harmful
effect. To allow no spending, the decision formulation needs an explicit
outside option or an upper-bound total budget.

On CPU and CUDA, the budget projection computes its intermediate arithmetic in
float64 to reduce rounding error. MPS does not support float64 and retains the
input precision. These are experimental outputs; do not use them for business
decisions without independent validation and suitable checks.

## Structured and mixed-geometry training

Use `--structured` with `train_controlled.py` to add correlated media spend
and persistent controls to the synthetic prior. `train_mixed.py` trains the
same direct-allocation architecture across four geometries: 2, 3, 4 and 6
channels with 52, 64 or 104 historical periods and 1, 2 or 3 controls. It
interleaves batches without padding and selects a checkpoint from separate
validation worlds. It still assumes additive one-period outcomes and does not
validate real-world use.

```sh
PYTHONPATH=src python examples/direct_budget/train_mixed.py \
  --output artifacts/direct-budget/mixed.safetensors
```

## Direct multi-period planning

`MultiPeriodBudgetModel` outputs a channel budget for each period. Inputs are
historical spend, outcome and controls, future period totals, and lower and
upper channel bounds for each period. The network returns the allocation
sequence only. It is trained with a synthetic multi-period outcome and
carryover; that training outcome is not an inference output.

```sh
PYTHONPATH=src python examples/direct_budget/train_planning.py \
  --baseline artifacts/direct-budget/structured.safetensors \
  --output artifacts/direct-budget/planning.safetensors
PYTHONPATH=src python examples/direct_budget/allocate_plan.py \
  --model artifacts/direct-budget/planning.safetensors --input plan-input.json
```

The JSON input uses `channels`, `spend`, `outcome` and `controls` as above,
plus `totals`, an array of period budgets. Optional `lower` and `upper` are
arrays of rows, one row per period and one value per channel. Each period
spends its total. The training worlds use the same channel-bound fractions in
every period; changing those fractions across periods is unvalidated. These
bounds do not express cross-period constraints such as
limits on week-to-week changes or a single budget cap across the horizon.
The example checkpoint is trained on four periods. Other horizons are not
validated. Observed controls are historical; future controls are not inputs.

`evaluate_planning_ols.py` compares direct planning with repeated one-period
budgets and an OLS rule. It also computes a Frank-Wolfe optimality-gap bound
for the synthetic concave response. The bound is conditional on the synthetic
mechanism, not an error guarantee on business data. See `EXPERIMENT.md` for
results, including a hidden-demand case where planning performed worse than the
one-period policy.

An alternative trainer, `train_planning_robust.py`, mixes four simulated hidden
shock strengths during training. Its filename does not imply general robustness
or identification of hidden confounding. Both planning checkpoints use the
same input and output format. `evaluate_planning_ood.py` provides a separate
linear carryover stress with observed controls and optional hidden demand.

`train_schedule.py` trains an alternative planning architecture that encodes
the entire future sequence of period totals and channel bounds. Its training
worlds vary the bound fractions across periods. It uses the same JSON format
through `allocate_plan.py`:

```sh
PYTHONPATH=src python examples/direct_budget/train_schedule.py \
  --baseline artifacts/direct-budget/structured.safetensors \
  --output artifacts/direct-budget/schedule.safetensors
```

This is also trained on four periods. The complete-schedule input permits future
limits to affect earlier decisions; it does not impose cross-period coupling
constraints or prove that the learned response to future limits is optimal.

## Limits relative to current budgets

Both JSON CLIs accept optional business limits in addition to `lower` and
`upper`:

- `upper_share`: maximum share of each period's total. Use `0.10` for 10%.
- `current`: the current amount in channel order, together with exactly one of:
  - `max_change`: maximum absolute change per channel in spend units.
  - `max_change_fraction`: maximum change relative to each current channel
    amount. `0.20` means at most 20% up or down.

Changes can be a scalar or one number per channel. Shares can be a scalar,
a channel vector, a name-to-share mapping, or for planning a period-by-channel
matrix. A mapping applies the same share in every period and defaults omitted
channels to 100%. Unknown channel names are rejected. Explicit
limits, share limits and change limits are intersected. The input is rejected
if the resulting limits cannot spend the required total. A relative change
from a zero current amount permits no increase; use an absolute change limit
when a zero-spend channel should be allowed to start.

For example, with four channels in TV, Search, Social, Video order, these fields
limit TV to 10% of each total and every channel to at most 10 spend units from
its current amount:

```json
{
  "current": [19, 30, 30, 21],
  "max_change": 10,
  "upper_share": {"TV": 0.10}
}
```

The single-period JSON and CSV allocation commands also accept
`"constraint_strategy": "project"`. This runs the model with no per-channel
limits and projects its resulting budget onto the requested limits using
Euclidean distance. The default, `"direct"`, passes the limits to the model
itself. Both return only the final budget. Projection does not establish that
the result maximizes the real outcome under the limits, and either strategy
may perform better for a particular dataset. The multi-period planning CLI
does not implement this option.

For multi-period planning the reference is the same `current` vector in every
period. It is not a maximum change between consecutive periods. That coupled
constraint is not implemented. The network still returns only budget amounts.

The model is projected into the combined feasible bounds. This guarantees the
specified bounds within floating-point tolerance; it does not prove that the
result is the best feasible allocation. In particular, limits unlike the
synthetic training scenarios need separate evaluation. Unknown JSON fields are
rejected so a misspelled constraint cannot silently disappear.

## Broader synthetic prior

`train_diverse.py` trains a one-period budget policy on a mixture of the
structured simulator and a broader simulator with correlated spend, time-varying
controls and demand, variable noise and positive, null or negative channel
effects. `evaluate_diverse.py` compares two saved policies on the same worlds
against a constrained OLS baseline. See [EXPERIMENT.md](EXPERIMENT.md) for
reproduction commands, measured tradeoffs and important limits. The policy
still emits only the allocation, not forecasts or contributions.

Constraints can also be applied after a free decision using a projection, but
that changes the decision and does not generally optimize the outcome within
the remaining feasible set. `evaluate_posthoc.py` compares that specific method
with supplying bounds to the policy from the outset; see the synthetic results
and limitations in [EXPERIMENT.md](EXPERIMENT.md).

`train_linear_mix.py` provides another experimental checkpoint trained on a
mixture of nonlinear and varied linear carryover worlds. Its comparison in
[EXPERIMENT.md](EXPERIMENT.md) shows improvements on the studied linear stress
and regressions on other simulated families, so it is not a general replacement.

## Using a dated CSV history

For local data, `allocate_csv.py` reads one observation per period. It requires
an ISO date column with strictly increasing, equally spaced calendar days,
finite numeric values, nonnegative spend and at least two rows. Missing or
irregular periods are rejected, never filled automatically. Monthly dates with
varying day counts are therefore not supported by this loader. Select the
channel columns in the intended allocation order, the aggregate outcome column
and any observed control columns explicitly. Units and the next decision period
must match the historical data and budget; the loader cannot verify those
business assumptions. No data are uploaded.

For example, given `history.csv`:

```csv
date,TV,Search,Sales,Promotion
2025-01-06,10,20,50,0
2025-01-13,12,18,55,1
```

and `budget.json`:

```json
{"budget": 100, "upper_share": {"TV": 0.10}}
```

run, after training a compatible checkpoint:

```sh
PYTHONPATH=src python examples/direct_budget/allocate_csv.py \
  --model artifacts/direct-budget/linear-mix.safetensors \
  --history history.csv --config budget.json \
  --channel TV --channel Search --outcome Sales --control Promotion
```

The output is a JSON total and channel allocations, without forecasts or
contributions. The two rows above demonstrate the file format only; such a
short history and this model's simulated training do not justify a business
decision. The JSON configuration supports the same bounds and change limits
as the direct JSON workflow. Missing controls must not be mistaken for
measured zero controls or evidence that confounding has been addressed.

## Experimental time-aware policy

`train_temporal.py` trains an alternative budget-only policy that supplies
one- and two-period lag associations to its observation encoder. It can
therefore respond to ordering of historical observations, unlike the previous
encoder's order-invariant aggregate apart from its final-period input. This
does not guarantee learned carryover is correct or improve decisions. It is not
a replacement for the other checkpoints; see [EXPERIMENT.md](EXPERIMENT.md).
The JSON and CSV allocation scripts accept its explicitly tagged checkpoint.

## Residual budget policy

`ResidualBudgetModel` freezes a temporal policy, projects its unconstrained
budget onto the supplied limits, and learns a channel-wise correction before
projecting again. Its zero-initialized correction starts at the projected base
allocation. At inference it returns only a feasible budget, not forecasts or
channel contributions. Training uses only the synthetic structured and diverse
worlds generated in this repository. The optional `--linear-replay` flag also
interleaves signed-linear synthetic worlds with varying budget bounds; checkpoint
selection then combines validation regret across the two prior families using
fixed weights. The independent `--interaction-replay` flag additionally
trains on synthetic positive and negative interactions between random channel
pairs and adds an interaction term to validation selection. Either flag or
both may be used. These choices do not establish performance on other mechanisms. It does not establish causal identification
or real-world budget performance. No trained checkpoint is bundled.

After training a temporal checkpoint as described above, train and use a
residual checkpoint:

```sh
PYTHONPATH=src python examples/direct_budget/train_residual.py \
  --base artifacts/direct-budget/temporal-matched.safetensors \
  --output artifacts/direct-budget/residual.safetensors
PYTHONPATH=src python examples/direct_budget/allocate_csv.py \
  --model artifacts/direct-budget/residual.safetensors \
  --history history.csv --config budget.json \
  --channel TV --channel Search --outcome Sales --control Promotion
```

The residual checkpoint includes its frozen base and is saved as safetensors.
The training script selects using separate synthetic validation worlds and
reports synthetic test regret and feasibility in an adjacent JSON file. It
trains on the repository's bounded one-period priors, not the private research
benchmarks or multi-period objectives. Evaluate new mechanisms and real
interventions before using its recommendations in practice.
