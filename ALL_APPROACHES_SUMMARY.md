# All Approaches Summary

## Approach 1 – Full Encoder+Decoder LoRA on Whisper-Small

**What we did**
- Model: Whisper-small (244M)
- LoRA: Full encoder + full decoder, rank 16, everything unfrozen
- Loss: Standard CE + extra weight on address tokens
- Data: ~100–200 multi-turn dialogues
- TTS: gTTS with phone-like corruption (band-pass + speed 0.9–1.1×)
- Audio quality: Intentionally noisy

**Key idea**
Try to "force" the model to learn addresses with full encoder–decoder LoRA on noisy synthetic data.

**Results**
WER −0.0532 (regression), Canonical Accuracy +0.11% (minimal), Token F1 ~0 (no improvement).

**Learnings**
Very small datasets cause overfitting. Full encoder–decoder LoRA on large models amplifies noise learning.

---

## Approach 2 – Full Encoder+Decoder LoRA on Whisper-Tiny

**What we did**
- Model: Whisper-tiny (39M) → smaller, easier to bend
- LoRA: Full encoder + full decoder, rank 16 (same as A1)
- Loss: Same as A1 (standard CE + address weighting)
- Data: ~600–700 multi-turn dialogues → more data than A1
- TTS: gTTS (Indian English), clean
- Audio quality: Clean, no artificial corruption

**Key idea**
Shrink the model, give it slightly more/cleaner data, keep everything else the same.

**Results**
WER +0.1060 (improvement), Canonical Accuracy +0.87% (improvement), Token F1 +0.0210 (improvement).

**Learnings**
Model size matters more than fancy tricks when data is small. Tiny model + clean data fine-tunes better than small model on noisy data.

---

## Approach 4 – Decoder-only LoRA + Sarvam TTS

**What we did**
- Model: Whisper-small (244M)
- LoRA: Encoder frozen, LoRA only on decoder, rank 32
- Loss: Plain CE, no address weighting
- Data: 2,500 single-sentence samples (not dialogues)
- TTS: Sarvam TTS (good quality)
- Split: 70/15/15
- Address mix: 40% with addresses, 60% without

**Key ideas**
- Freeze encoder to keep speech features intact
- Use higher LoRA rank on decoder to compensate
- Move from dialogues to single sentences
- Switch from low-quality gTTS to high-quality Sarvam

**Results**
WER −0.0416 (regression), Canonical Accuracy −20.67% (severe drop), Token F1 +0.1461 (best token-level improvement).

**Learnings**
Freezing the encoder + cranking up decoder LoRA improves token recognition but hurts exact address accuracy. Good phonemes ≠ good full addresses. Single sentences also give less context.

---

## Approach 5 – Curriculum + Partial Encoder LoRA + Phoneme Variants

**What we did**
- Model: Whisper-small (244M)
- LoRA: Partial encoder (last block, rank 8) + decoder rank 16
- Loss: CE + ramped address weighting (1.2× → 2.0× over training)
- Data: 2,500 samples (2,000 used for training), natural utterances
- TTS: Sarvam, multi-speaker
- Training: Curriculum – stage 1 phoneme-faithful, stage 2 canonical
- Extras: 5–10 phoneme variants per address, mild SpecAugment, random address position

**Key ideas**
- Only touch last encoder block (keep most pretraining intact)
- Two-stage training: first nail pronunciations, then canonical text
- Gradually increase importance of address tokens
- Add lots of phoneme and speaker diversity

**Results**
WER +0.0883 (improvement), Canonical Accuracy −12.60% (drop), Token F1 +0.1324 (improvement).

**Learnings**
Partial encoder LoRA stabilizes training and improves WER, but the curriculum and phoneme focus confuse the model on final canonical addresses. Robust to variations, still weak on exact match.

---

## Approach 6 – Focal Loss + Sequence Bonus, No Curriculum

**What we did**
- Model: Whisper-small (244M)
- LoRA: Partial encoder (last block, rank 8) + decoder rank 16 (q/k/v, added k_proj)
- Loss:
  - Focal loss (gamma 2.0)
  - Sequence-level bonus (0.5) for fully correct outputs
  - Label smoothing 0.1
- Data: Random 2,500 picked from a 10,000-sample pool, augmented to 6,250 (2.5×)
- TTS: Sarvam, multi-speaker
- Training: No curriculum, train directly on canonical text
- Augmentation: Speed, pitch, speakers, position, phoneme variants

**Key ideas**
- Use focal loss to focus on "hard" tokens (addresses, tricky parts)
- Add a sequence-level reward for perfect predictions
- Remove curriculum to avoid confusing the target form
- Work on a randomly sampled 2.5K subset from 10K so iterations stay fast

**Results**
WER +0.1965 (best WER improvement), Canonical Accuracy −5.40% (small drop), Span F1 0.4808 (improvement).

**Learnings**
Focal loss + sequence bonus fix a lot of issues from earlier approaches. Accuracy drop is much smaller now, even with a subset of data. Random 2.5K from 10K gives enough diversity for good WER while keeping experiments cheap and quick.

---

## Approach 7 – Whisper-Base + Focal Loss (Final) ⭐

**What we did**
- Model: Whisper-base (74M) → ~3.3× smaller than Whisper-small
- LoRA: Partial encoder (last block, rank 8) + decoder rank 24
- Loss: Same as A6 – focal loss (gamma 2.0) + sequence bonus 0.5 + label smoothing 0.1
- Data: Same as A6 – random 2,500 from 10K pool, augmented to 6,250
- TTS: Sarvam, multi-speaker
- Training: 3 epochs, batch size 8 (vs 2 earlier), BF16
- Compute: ~12 minutes, 8GB GPU (T4)

**Key ideas**
- Swap Whisper-small → Whisper-base (smaller, but still capable)
- Increase decoder LoRA rank to 24 so the smaller model still has enough capacity
- Bump batch size since the model is lighter
- Keep the loss recipe and data strategy from A6 the same

**Results**
| Metric | Value |
|--------|-------|
| WER | +0.0438 (improvement) |
| Canonical Accuracy | +4.40% (only approach with accuracy gain) |
| Token F1 | +0.0274 (improvement) |
| Span F1 | 0.4870 (best) |
| Phoneme-aware WER | 0.0907 (improvement) |

**All metrics improved.**

**Learnings**
This is the only setup where accuracy actually goes up, and all other metrics also improve. Whisper-base is easier to steer with LoRA than Whisper-small, and the 2.5K-from-10K sampling is enough to generalize without running full 10K every time. **This is the version worth deploying.**
