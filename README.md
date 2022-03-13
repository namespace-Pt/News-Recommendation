### Instruction
1. Download MIND dataset [here](https://msnews.github.io/)
2. Save MIND dataset in a directory, e.g. `~/Data`
3. ```
   pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```
5. ```bash
   python -m main.twotower --data-root ~/Data   \
                           --batch-size 8       \
                           --world-size 2       \
                           --news-encoder bert
   ```
   - `--world-size` defines number of gpus in ddp training
   - `--news-encoder` defines news encoder
   - more parameters can be found in [src/utils/manager.py](src/utils/manager.py)