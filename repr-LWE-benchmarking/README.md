# Obtaining Cool and Cruel measurements

In order to compare the primal attack against Cool and Cruel, we provide scripts for automatinng the latter attack using the code release from 2024/1229.
Overall you will need:
- Code from the 2024/1229 release
- A specific Conda environment
- A machine with a cuda-compatible graphics card
- The flatter executable in your $PATH

To obtain the code and check you satisfy all dependencies or get help obtaining them, run
```bash
bash fetch.sh
```

Once all checks pass, run the copy of `reproducing.sh` inside the `LWE-benchmarking` directory created by `fetch.sh` within this directory.
This will run C&C experiments and save the timings. Run `reproducing.sh -h` to see various attack options.
