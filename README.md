# To-do
## Coding
- [x] Check step up
- [x] Check data builder
- [x] Check model builder
- [x] Run a small experiment to know pipeline (data input, data loader, build model, train model, evaluate)
- [ ] Check SSFA
- [ ] Check VoTr
- [x] Add IA-SSD
- [ ] Create Models (mix modules)

## Writing
- [x] Introduction
- [ ] Related Work

## Commands

### Training
```bash
torchrun --nproc_per_node <number gpus> train.py <configuration path> <output path> --batch_size <batch size per gpu> --num_workers ?
```
