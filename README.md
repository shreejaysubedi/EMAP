# EMAP Dataset Challenge — Arousal Regression

**Submitter:** Shreejay Subedi
**Mentor:** Dr. Saurav K. Aryal

This submission covers the three regression targets in the challenge — arousal,
heart rate, and GSR — each predicted separately, plus the bonus binary
classification of arousal. Arousal is handled by an ensemble of CNN + Bi-LSTM
networks that read EEG scalp topomaps; heart rate and GSR are handled by
feature-selected gradient-boosted models. All of it runs from `prediction.py`,
which writes `Pred_Arousal`, `Pred_HR`, and `Pred_GSR`.

## Results on the validation set

We evaluated on the 29 held-out validation participants (696 trials, 84,019
half-second bins). All numbers below are reproducible end to end from
`prediction.py`.

Regression (RMSE, with the target's own standard deviation for reference):

| Target | RMSE | Target std | Notes |
|---|---|---|---|
| Arousal | 0.3055 | 0.323 | CNN + Bi-LSTM ensemble |
| Heart rate (BPM) | 9.42 | 9.41 | feature-selected LightGBM |
| GSR | 1.03e-5 | 1.02e-5 | feature-selected LightGBM |

Arousal is well below the "predict-the-mean" level (std 0.323). Heart rate and
GSR sit close to it: the 0.5-second band-power averages keep very little of the
fast pulse and skin-conductance dynamics, so once the target itself is removed
from the inputs there is not much left to predict them from. We report them
honestly rather than dressing them up.

To be sure HR and GSR were not just under-explored, we ran a broad search for
each: four algorithm families (ridge, random forest, LightGBM, a small MLP)
crossed with five feature views (EEG only, peripherals + position, a
physiology-aware subset with respiration / blood-volume and their lags, all
features, and all features plus temporal lags) — about 34 models in total. The
results are in `hr_gsr_exploration_results.json`. The best heart-rate model
explained only 1.7% of the variance (R² = 0.017), and the best GSR model
explained none (R² ≤ 0). No algorithm or feature view broke the floor, which is
what we would expect if the signal simply is not present in these summaries.

Bonus classification of arousal (low ≤ 0.5 vs high > 0.5), by thresholding the
arousal regression at 0.5:

| Metric | Value |
|---|---|
| F1 (binary) | 0.6609 |
| F1 (macro) | 0.5980 |
| Accuracy | 0.6020 |

## How the model works

Each row in the data is one 0.5-second bin: 64 EEG channels with 4 band-power
values each, plus 4 peripheral signals (heart rate, GSR, blood volume,
respiration). For every bin we take the 64×4 EEG values and arrange them as a
24×24 picture of the scalp, using the standard 10-20 electrode positions from
mne-python with inverse-distance interpolation between the electrodes. A small
2-D CNN reads that picture. At the same time, an MLP reads the raw 260 numbers
directly, together with a few summary features: per-band statistics across
channels, short differences of the peripheral signals, and where the bin sits
inside its trial. The two streams are added and passed through a 2-layer
bidirectional LSTM so the model can use the time context within a trial.

The prediction we submit is a weighted average of three trained models, with
weights fit on the validation set:

| Model | Topomap | Predicts | Weight |
|---|---|---|---|
| `model.pt` | 24×24 | arousal | 0.37 |
| `model_quartile.pt` | 24×24 | arousal | 0.32 |
| `model_q48dev.pt` | 48×48 | arousal minus the trial average | 0.32 |

Two details are worth calling out. `model_quartile` adds a feature for which of
the four video repetitions ("loops") a bin belongs to. The third model is
trained to predict only how arousal moves within a trial, after subtracting that
trial's average, so it can specialize in the moment-to-moment shape; at
prediction time we add the training-set average back. That model also uses a
larger 48×48 topomap, which the weight search preferred over the 24×24 version
of the same model. After averaging the three predictions, we lightly smooth each
trial (Gaussian, sigma 0.5 bins), apply a small variance correction, and clip to
the [0, 1] range.

## Feature selection

We used two approaches and compared them, one for each target (arousal, HR, GSR):

- A filter method: a univariate F-test (`f_regression`) that scores each feature
  by how strongly it relates to the target on its own.
- An embedded method: LightGBM gain importance from a model trained on all
  features, which also captures interactions between features.

We keep the features the gradient-boosted model actually splits on (gain above
zero, capped at the top 80), then retrain a smaller model on just that subset.
`selected_features.csv` lists every feature with its gain and a selected flag for
each target. When predicting heart rate we drop `heartrate_mean` from the inputs,
and when predicting GSR we drop `GSR_mean`, so a target is never used to predict
itself.

Selection roughly halves or quarters the feature count and does not hurt
accuracy; for arousal it helped a little (all features 0.3158, selected 80
features 0.3123). Counts kept: arousal 80, heart rate 43, GSR 80.

The arousal *ensemble* (our best arousal model) uses all channels, since the CNN
already learns which scalp regions matter from the topomap; the explicit feature
selection above is what produced `selected_features.csv` and the HR/GSR models.

## What we tried

We ran a set of single-change experiments on a fixed baseline (same seed, same
number of epochs) to see what actually moved the validation RMSE:

| Change | Val RMSE | vs baseline |
|---|---|---|
| baseline | 0.3126 | — |
| loop-quartile feature | 0.3045 | −0.008 |
| larger 48×48 topomap | 0.3077 | −0.005 |
| peripherals added as topomap channels | 0.3129 | no change |
| spatial Laplacian (CSD) maps | 0.3133 | no change |
| stimulus-ID embedding | 0.3171 | worse, overfits |
| all changes together | 0.3185 | worse |

The loop-quartile feature and the larger topomap were the two that helped, and
both ended up in the final model. The rest either did nothing or made the model
overfit. Notably, turning everything on at once was worse than any single
change, which suggests most of those extra features were adding noise rather
than new signal.

## Where the error comes from

It is useful to split the error in two. One part is getting each trial's overall
arousal level right (the trial mean). The other is tracking how arousal rises and
falls within a trial. These combine roughly as
RMSE² ≈ (trial-mean error)² + (within-trial error)².

| Part of the error | Our model | Best we reached |
|---|---|---|
| Trial mean (overall level) | 0.264 | about 0.264 |
| Within-trial (shape) | 0.155 | about 0.154 |

The within-trial part is close to as good as we could get on this data. Larger
models, more seeds, and ensembling did not push it much under 0.154.

The trial-mean part is harder, and it is where most of the error lives. We tried
several ways to predict a trial's average arousal: a constant, ridge and
gradient-boosted models on trial-level summaries, a per-subject context model
built from each participant's other trials, and a larger transformer that looks
at all 24 of a participant's trials at once. They all landed around 0.27, which
is basically what you get by guessing the global average. As a sanity check, if
we give the model the true trial means and keep our own within-trial
predictions, the RMSE falls to about 0.15. So almost all of the remaining error
is in the trial-mean term, and it does not look recoverable from these features
when the validation people were never seen in training.

Our interpretation is that the absolute arousal level of an unseen person is
largely subject-specific, and the cues for it seem to sit in the raw EEG
waveform rather than in the band-power summaries. Each 0.5-second window
compresses 250 raw samples per channel into four band-power numbers, and a lot
of the slower, person-specific structure is averaged out in that step.

## Files

| Path | What it is |
|---|---|
| `EMAP_submission.pptx` | Slide deck: approach and results |
| `prediction.py` | End-to-end inference, including all preprocessing |
| `model.py`, `model_v2.py` | Model definitions used by `prediction.py` |
| `selected_features.csv` | The features the model uses |
| `model_assets/model*.pt` | The three arousal models (CNN + Bi-LSTM) |
| `model_assets/HR_lgbm.txt`, `GSR_lgbm.txt` | Feature-selected LightGBM models for heart rate and GSR |
| `model_assets/arousal_lgbm.txt` | Feature-selected LightGBM for arousal (used for the feature-selection study) |
| `model_assets/*.norm.npz` | Per-model normalisation statistics |
| `model_assets/*.idw.npz` | Per-model topomap interpolation kernel (24×24 and 48×48) |
| `model_assets/blend_weights.json` | Ensemble weights, calibration factor, smoothing |
| `training_code/` | Scripts to rebuild the models, including `feature_selection.py` and `hr_gsr.py` |
| `prediction_plots/` | Predicted-vs-true curves and parity plots for arousal, HR, and GSR |

## Running inference

```bash
pip install -r requirements.txt
python prediction.py --input <folder_or_single_csv> --output predictions.csv
```

The output has columns `Participant, Trial, Bin, Pred_Arousal, Pred_HR,
Pred_GSR`. Arousal is in [0, 1]; heart rate is in BPM; GSR is in its recorded
unit.

## Reproducing the models

```bash
python training_code/build_cache.py        # cache the CSVs into numpy arrays
python training_code/train.py        ...    # base arousal model
python training_code/train_dev.py    ...    # within-trial model
python training_code/train_v2.py --use_quartile 1 ...   # loop-quartile model
python training_code/train_v2_dev.py --use_quartile 1 --grid 48 ...  # 48x48 within-trial model
python training_code/final_ensemble_v2.py   # fit the ensemble weights on validation
```
