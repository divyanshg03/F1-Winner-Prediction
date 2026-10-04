My F1 prediction model was 95.7% accurate.

A model that predicts "nobody wins, ever" was 95.0%.

I built that project in 2025. This year I audited it properly, rebuilt it, and tested it the hard way: 188 real races, retraining before every race so it never sees the future.

What I found:

→ A year of past form was worth about the same as just knowing the starting grid
→ The full model's probabilities are genuinely better (+0.30 nats/race, 95% CI 0.14–0.47) and well calibrated
→ But its accuracy edge over "pole position wins" (59% vs 53%) is NOT statistically significant
→ And the edge nearly vanished in 2024–26. It only looked strong while one team dominated

Along the way I found my own bugs: a feature that was always zero, a Transformer training on NaN loss, and a test set that overlapped with training.

The part I'm proudest of is the leak test. It shuffles a race's own result, rebuilds the features, and asserts nothing changed.

I then threw everything else at it: circuit geometry, tyre proxies, practice pace, weather, a lap-by-lap race simulator. Nothing beat the simple model. Against a betting market (Polymarket, 17 races) it was a statistical tie.

I also froze the next race prediction in the repo before lights-out, so it can be scored publicly, right or wrong.

The lesson: optimise for being hard to fool, not for the number that looks good.

Full write-up, code and one-command reproduction in the comments.

#MachineLearning #DataScience #Formula1 #Forecasting
