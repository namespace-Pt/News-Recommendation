### Instruction
1. Download MIND dataset [here](https://msnews.github.io/)
2. Save MIND dataset in a directory, e.g. `~/Data`
3. ```
   pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```
5. ```
   python -m main.gateformer --world-size 2 --data-root ~/Data
   ```