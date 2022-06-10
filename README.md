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
torchrun --nproc_per_node <number_gpus> train.py <configuration_path> <output_path> --batch_size <batch_size_per_gpu> --num_workers ?
```

### Testing
```bash
 torchrun --nproc_per_node <number_gpus> evaluate.py <configuration_path> <checkpoint> <output_path> --batch_size <batch_size_per_gpu> --num_workers ?
```
