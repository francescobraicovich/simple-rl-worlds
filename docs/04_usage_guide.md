# Full Usage Guide & Examples

This guide provides practical, end-to-end command sequences for running the primary experimental workflows.

---

### Prerequisites

1.  **Installation:** Ensure you have installed all dependencies from `requirements.txt`.
2.  **Configuration:** All scripts use the `config.yaml` file for parameters. Make sure it is configured for your desired environment and model settings.

---

## Workflow 1: The JEPA Approach

This workflow uses self-supervised training to learn representations and then evaluates them.

**Step 1: Collect Data**

First, generate a dataset of experiences. This only needs to be done once per environment.

```bash
python src/scripts/collect_load_data.py
```

**Step 2: Train the JEPA Encoder and Predictor**

This is the core self-supervised training step.

```bash
python src/scripts/train_jepa.py
```
*This will save the best `encoder` and `predictor` models to `weights/jepa/`.*

**Step 3 (Optional): Train a Decoder for Evaluation**

To visualize the representations learned by JEPA, train a decoder on top of the frozen models.

```bash
python src/scripts/train_jepa_decoder.py
```
*This uses the weights from `weights/jepa/` and saves the trained decoder to `weights/jepa_decoder/`.*

**Step 4: Train Reward Predictors**

Finally, evaluate how useful the learned representations are for a downstream task like reward prediction.

```bash
# Train a reward predictor using the ground-truth next state
python src/scripts/train_reward_predictor.py --approach jepa

# Train a reward predictor using the model's *predicted* next state
python src/scripts/train_dynamics_reward_predictor.py --approach jepa
```

---

## Workflow 2: The Encoder-Decoder Approach

This workflow uses a more traditional, end-to-end reconstruction-based method.

**Step 1: Collect Data**

If you haven't already, generate the dataset.

```bash
python src/scripts/collect_load_data.py
```

**Step 2: Train the Encoder, Predictor, and Decoder Jointly**

This single script trains all three components with a reconstruction loss.

```bash
python src/scripts/train_encoder_decoder.py
```
*This will save the best `encoder`, `predictor`, and `decoder` models to `weights/encoder_decoder/`.*

**Step 3: Train Reward Predictors**

Evaluate the learned representations from the encoder-decoder model.

```bash
# Train a reward predictor using the ground-truth next state
python src/scripts/train_reward_predictor.py --approach encoder_decoder

# Train a reward predictor using the model's *predicted* next state
python src/scripts/train_dynamics_reward_predictor.py --approach encoder_decoder
```
