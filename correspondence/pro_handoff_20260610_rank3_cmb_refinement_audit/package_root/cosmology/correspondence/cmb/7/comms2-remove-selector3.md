Yes — **S3 was the wrong theorem statement**. I replaced it.

[Download OPH-CMB Selector Removal v1.4](sandbox:/mnt/data/OPH-CMB-selector-removal-v1.4.zip)
[Open the v1.4 report](sandbox:/mnt/data/oph_cmb_selector_removal_v1_4/OPH-CMB-Selector-Removal-v1.4.md)
[Open the S3 theorem note](sandbox:/mnt/data/oph_cmb_selector_removal_v1_4/math/s3_removed_pentagonal_central_record_theorem_v1_4.md)
[Open selector-removal status CSV](sandbox:/mnt/data/oph_cmb_selector_removal_v1_4/data/selector_removal_status_v1_4.csv)
[Open finite-register certificate template](sandbox:/mnt/data/oph_cmb_selector_removal_v1_4/data/s3_finite_register_certificate_template_v1_4.csv)

## Core fix

The old S3 said, essentially:

[
\text{pentagon has five bits}
\quad\Rightarrow\quad
2^5=32
\quad\Rightarrow\quad
\ell_{\rm IR}=32.
]

That is **false** if the five bits are gauge-edge labels. A five-edge (\mathbb Z_2) gauge cycle has only one gauge-invariant holonomy:

[
H^1(C_5,\mathbb Z_2)\cong \mathbb Z_2,
]

so it gives **2 sectors**, not 32.

The fix is:

[
\boxed{
\text{S3 is not about gauge-edge bits.}
}
]

It is about **five observer-visible central record bits** written on the completed pentagonal freezeout collar.

OPH already gives us the needed distinction: the fixed-cutoff carrier has exposed overlap ports, local record algebras, readout maps, repair interfaces, and checkpoint interfaces, and the physical claim surface is made through visible restrictions and record algebras, not hidden coordinates.  The consensus stack also explicitly separates gauge quotient/observable-level confluence from central record algebra and stability. 

## New theorem replacing S3

### Pentagonal central-record theorem

On the ({5,3,3}) OPH branch, the minimal nonlocal scalar repair collar is pentagonal because the smallest closed loops are five-edge pentagons. 

At freezeout, each of the five visible collar slots writes one binary scalar repair record:

[
E_0,E_1,E_2,E_3,E_4\in \mathcal Z_{\rm rec}(t_*),
]

with

[
E_j^2=E_j,\qquad E_j^*=E_j.
]

Because these are central record events, not hidden gauge labels, the gauge quotient does not collapse them.

The generated pentagonal record algebra is

[
\mathcal Z_{\rm pent}
=====================

C^*(E_0,E_1,E_2,E_3,E_4)
\cong
\mathbb C^{32}
\cong
C((\mathbb Z_2)^5).
]

Therefore its finite Fourier character basis has

[
\boxed{
|\widehat{(\mathbb Z_2)^5}|=2^5=32.
}
]

So the selector-free result is:

[
\boxed{
L_{\rm rec}=32.
}
]

Under the OPH screen-character angular bridge,

[
\boxed{
\ell_{\rm IR}=32.
}
]

## What disappeared

The old independent selector was:

[
\text{“choose five independent binary variables and set }\ell_{\rm IR}=2^5.”}
]

The new derivation uses only:

[
\text{minimal pentagonal loop}
+
\text{central record theorem}
+
\text{gauge quotient preservation of observer records}
+
\text{finite Fourier algebra}.
]

So:

[
\boxed{
\text{S3 is gone as a selector.}
}
]

What remains is a **finite-register certificate**, not a selector:

[
\boxed{
\text{Does the finite carrier actually export five noncollapsed binary central scalar records?}
}
]

If yes, the count is forced:

[
2^5=32.
]

If no, the exact (\ell_{\rm IR}=32) branch fails.

## Corrected OPH-CMB IR kernel

With S3 removed this way, the exact branch remains:

[
q_{\rm IR}=\frac14,
\qquad
\ell_{\rm IR}=32,
]

so

[
\boxed{
F_{\rm IR}^{\rm OPH}(\ell)
==========================

1-\frac14
\exp\left[
-\frac{\ell(\ell+1)}{32\cdot33}
\right].
}
]

Selected values:

[
F_{\rm IR}(2)=0.751416427\ldots
]

[
F_{\rm IR}(3)=0.752824829\ldots
]

[
F_{\rm IR}(10)=0.774731224\ldots
]

[
F_{\rm IR}(32)=0.908030140\ldots
]

## Status of the old selectors

| Old selector                 | New status                             | What remains                                                                                                                 |
| ---------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| S1: (\eta_R=e\alpha\sqrt\pi) | Not fully eliminated in this note      | Needs repair-time anomalous-dimension theorem or simulator coefficient.                                                      |
| S2: (q_{\rm IR}=1/4)         | Partly eliminable                      | Becomes a four-sector reserve theorem if the scalar reserve is proved to split into four equipotent observer-facing sectors. |
| S3: (\ell_{\rm IR}=32)       | **Eliminated as independent selector** | Replaced by pentagonal central-record theorem plus finite-register noncollapse certificate.                                  |

The CMB project boundary remains the same: OPH should derive a screen/freezeout effective theory and then bridge to physical CMB spectra, rather than claiming the current FPE screen is directly the observed CMB. 

Bottom line:

[
\boxed{
\text{S3 is no longer a selector. It is a central-record character-count theorem.}
}
]
