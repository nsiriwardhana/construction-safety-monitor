from ultralytics import YOLO
from rule_engine import classify_worker


def analyze_scene(image_path, model_path="../models/best.pt", conf=0.25):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, save=False)
    result = results[0]

    person_boxes = []
    helmet_boxes = []
    vest_boxes = []

    for box in result.boxes:
        cls = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        label = model.names[cls]

        if label == "person":
            person_boxes.append(xyxy)
        elif label == "helmet":
            helmet_boxes.append(xyxy)
        elif label == "vest":
            vest_boxes.append(xyxy)

    safe_count = 0
    unsafe_count = 0
    worker_reports = []

    for i, person_box in enumerate(person_boxes, start=1):
        status, missing = classify_worker(person_box, helmet_boxes, vest_boxes)

        if status == "safe":
            safe_count += 1
        else:
            unsafe_count += 1

        worker_reports.append({
            "worker_id": i,
            "status": status,
            "missing": missing
        })

    scene_status = "SAFE" if unsafe_count == 0 else "UNSAFE"

    return {
        "workers_detected": len(person_boxes),
        "safe_workers": safe_count,
        "unsafe_workers": unsafe_count,
        "scene_status": scene_status,
        "worker_reports": worker_reports
    }


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    report = analyze_scene(image_path)

    print("\nScene Summary")
    print("--------------------")
    print(f"Workers detected: {report['workers_detected']}")
    print(f"Safe workers: {report['safe_workers']}")
    print(f"Unsafe workers: {report['unsafe_workers']}")
    print(f"Scene status: {report['scene_status']}")

    print("\nWorker Reports")
    print("--------------------")
    for worker in report["worker_reports"]:
        if worker["status"] == "safe":
            print(f"Worker {worker['worker_id']}: SAFE")
        else:
            print(
                f"Worker {worker['worker_id']}: UNSAFE "
                f"(missing {', '.join(worker['missing'])})"
            )