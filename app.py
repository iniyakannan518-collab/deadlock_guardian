
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.resource_system import ResourceSystem
from src.detection import cycle_detect, detect_deadlock_multi_instance

app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "online",
        "service": "Deadlock Guardian API"
    })


@app.post("/api/safety")
def safety():
    try:
        data = request.get_json(force=True)

        system = ResourceSystem(
            total=data["total"],
            allocation=data["allocation"],
            max_claim=data["max_claim"],
            process_names=data.get("process_names"),
            resource_names=data.get("resource_names")
        )

        safe, sequence = system.is_safe_state()

        return jsonify({
            "success": True,
            "safe": safe,
            "safe_sequence": sequence,
            "available": system.available,
            "need": system.need()
        })

    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


@app.post("/api/request")
def resource_request():
    try:
        data = request.get_json(force=True)

        system = ResourceSystem(
            total=data["total"],
            allocation=data["allocation"],
            max_claim=data["max_claim"],
            process_names=data.get("process_names"),
            resource_names=data.get("resource_names")
        )

        granted, reason, sequence = system.request_resources(
            int(data["process_index"]),
            data["request"],
            verbose=False
        )

        return jsonify({
            "success": True,
            "granted": granted,
            "reason": reason,
            "safe_sequence": sequence,
            "available": system.available,
            "allocation": system.allocation,
            "need": system.need()
        })

    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


@app.post("/api/detect")
def detect():
    try:
        data = request.get_json(force=True)

        allocation = data["allocation"]
        resource_request_matrix = data["request"]

        deadlocked, cycle, wait_for_graph = cycle_detect(
            __import__("src.rag", fromlist=["build_rag"]).build_rag(
                allocation,
                resource_request_matrix
            )
        )

        return jsonify({
            "success": True,
            "deadlocked": deadlocked,
            "cycle": cycle,
            "wait_for_edges": [
                [str(source), str(target)]
                for source, target in wait_for_graph.edges()
            ]
        })

    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


if __name__ == "__main__":
    print("Deadlock Guardian API starting...")
    print("Open the dashboard separately in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
