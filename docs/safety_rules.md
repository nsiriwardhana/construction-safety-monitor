# Safety Rules Definition

## Overview

This system evaluates construction site safety by detecting workers and identifying whether they are wearing required personal protective equipment (PPE). The system uses object detection to locate workers and PPE items, followed by a rule-based compliance engine that determines whether each worker is compliant with safety requirements.

The goal of the system is to automatically determine whether a construction scene is **safe or unsafe**.

---

## Defined Safety Rules

The following safety rules are implemented in the system.

### Rule 1 – Worker Must Wear a Helmet

Every detected worker must wear a safety helmet.

A violation occurs when:

* A worker is detected but no helmet is detected within the worker’s bounding box.

Example violation:

* Worker visible without a helmet.

---

### Rule 2 – Worker Must Wear a High-Visibility Vest

Every detected worker must wear a high-visibility safety vest.

A violation occurs when:

* A worker is detected but no vest is detected within the worker’s bounding box.

Example violation:

* Worker present but not wearing a safety vest.

---

## Worker Compliance Logic

For each detected worker, the system checks whether a helmet and vest are present.

Worker status is determined using the following logic:

* **SAFE** → Helmet AND vest detected.
* **UNSAFE** → Helmet missing OR vest missing.

If multiple PPE items are missing, the system reports all missing equipment.

Example outputs:

```
Worker 1: SAFE
Worker 2: UNSAFE (missing vest)
Worker 3: UNSAFE (missing helmet)
```

---

## Scene-Level Safety Evaluation

The system aggregates worker-level results to determine the safety status of the entire scene.

Scene status is determined using the following rule:

* **SAFE SCENE** → All detected workers are compliant.
* **UNSAFE SCENE** → One or more workers violate safety rules.

Example scene summary:

```
Workers detected: 3
Safe workers: 1
Unsafe workers: 2
Scene status: UNSAFE
```

---

## Violation Reporting

The system generates human-readable safety reports that identify:

* Number of workers detected
* Number of compliant workers
* Number of violations
* Missing PPE items

This approach makes the system suitable for automated safety monitoring applications.

---

## Known Limitations

The current rule-based system has several limitations:

* PPE detection accuracy depends on the quality of the trained model.
* Occluded workers may reduce detection accuracy.
* The system currently checks only helmets and vests.
* Worker-to-PPE association uses bounding box containment, which may occasionally produce incorrect matches in crowded scenes.

Future improvements may include additional safety rules and more advanced worker-tracking methods.


