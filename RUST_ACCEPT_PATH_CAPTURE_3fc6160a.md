# Rust accept-path capture @3fc6160a

- crate tip SHA (pre-repair baseline): `3fc6160a0f5f2be8df6349e0a1070b2c55958897`
- inputs: tiny two-tier FIPC fixture (n_persons=60, Qp=Qs=7, max_iter=8)
- prior init: mean=0, cov=I
- n_iter=8, n_accepted=8, n_rollback_full=0
- termination: max_iter_reached
- selected decision: `iter=1;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3751378502e2;fixed_ll=-7.3751378502e2;scale_ll=-7.3428832892e2;scale_pd=true;mean=[-0.07459283003934392, -0.25210721473719855]`
- root cause (Rust path): recovery tried scale before mean and capped mean_alpha at 0.1, so remapped-LL-improving mean steps could not take a full posterior-moment step; pre-pairing also skipped mean entirely once scale set accepted=true.
- all decisions:
  - `iter=0;branch=joint_full_accept;ll=-7.3796328377e2;cand_ll=-7.3751378502e2;mean=[-0.026047795723652456, -0.0770490317794472]`
  - `iter=1;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3751378502e2;fixed_ll=-7.3751378502e2;scale_ll=-7.3428832892e2;scale_pd=true;mean=[-0.07459283003934392, -0.25210721473719855]`
  - `iter=2;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3428832892e2;fixed_ll=-7.3428832892e2;scale_ll=-7.3250595763e2;scale_pd=true;mean=[-0.10019447548372654, -0.37400514542849334]`
  - `iter=3;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3250595763e2;fixed_ll=-7.3250595763e2;scale_ll=-7.3143328605e2;scale_pd=true;mean=[-0.11369609350495669, -0.4597789466920486]`
  - `iter=4;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3143328605e2;fixed_ll=-7.3143328605e2;scale_ll=-7.3072258998e2;scale_pd=true;mean=[-0.12083279012565457, -0.5207133309078493]`
  - `iter=5;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3072258998e2;fixed_ll=-7.3072258998e2;scale_ll=-7.3020340730e2;scale_pd=true;mean=[-0.12464050256998065, -0.5643761458802808]`
  - `iter=6;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3020340730e2;fixed_ll=-7.3020340730e2;scale_ll=-7.2978981869e2;scale_pd=true;mean=[-0.12672379483807697, -0.5959065858919496]`
  - `iter=7;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.2978981869e2;fixed_ll=-7.2978981869e2;scale_ll=-7.2943746275e2;scale_pd=true;mean=[-0.12792664578036458, -0.6188359703851906]`
