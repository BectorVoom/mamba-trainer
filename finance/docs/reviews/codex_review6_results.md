Prior-review status:

1. Fixed  
2. Fixed  
3. Fixed  
4. Fixed  
5. Fixed  
6. Fixed  
7. Fixed  
8. Fixed  
9. Fixed  
10. Fixed  
11. Fixed  
12. Fixed  
13. Fixed  
14. Fixed  
15. Fixed  
16. Fixed  

Remaining problems:

| Location | Severity | What is wrong | Fix |
|---|---|---|---|
| [RESULTS.md:27](/home/user/Documents/workspace/mamba-trainer/finance/RESULTS.md:27) | major | E_mamba_supcon is included in “Every Mamba variant … significantly lower,” but its buy-and-hold Sharpe interval is −0.14 [−0.29, +0.02], which includes zero. | Exclude E, or state its interval includes zero. |
| [RESULTS.md:295](/home/user/Documents/workspace/mamba-trainer/finance/RESULTS.md:295) | minor | “0.011 below it” matches neither reported prior exactly: 1.0767 − 1.0663 = 0.0104; 1.0794 − 1.0663 = 0.0131. | Name the prior and use the exact/appropriately rounded difference. |
| [RESULTS.md:393](/home/user/Documents/workspace/mamba-trainer/finance/RESULTS.md:393) | minor | Auto does not switch to chunked solely below 8,192 channels: code also requires `T > 2 × segment_length`. | Say Auto selects chunked when `B×D < 8,192` **and** the sequence exceeds two segments. |
| [RESULTS.md:398](/home/user/Documents/workspace/mamba-trainer/finance/RESULTS.md:398) | major | The supplied benchmark logs do not substantiate the 37%, register-spill cause, 6/19 ms, or 99/304 ms profile/A-B figures. | Publish the profiler and A/B benchmark logs, or remove these claims. |
| [RESULTS.md:419](/home/user/Documents/workspace/mamba-trainer/finance/RESULTS.md:419) | minor | The 2.2–2.6 minute per-fold figure is not in the supplied benchmark logs. | Cite/publish its timing log or remove it. |
| [RESULTS.md:353](/home/user/Documents/workspace/mamba-trainer/finance/RESULTS.md:353) | major | The CUDA model-test count is explicitly unsupported by a retained log, despite the table being presented as final-run results. | Publish the CUDA test log, or omit that table cell/claim. |
