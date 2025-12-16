from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, redirect, request, url_for

from .moderation.service import ReviewService


SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "AI Review Service API",
        "description": "Hybrid moderation service that blends ML predictions with rules.",
        "version": "1.0.0",
    },
    "basePath": "/",
}


def register_routes(app: Flask, service: ReviewService) -> None:
    @app.route("/", methods=["GET"])
    def index() -> Any:
        return redirect(url_for("flasgger.apidocs"))

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.route("/review", methods=["POST"])
    def review() -> Any:
        """
        Submit a review text for moderation.
        ---
        tags:
          - Review
        consumes:
          - application/json
        parameters:
          - in: body
            name: payload
            required: true
            schema:
              type: object
              required:
                - comment
              properties:
                comment:
                  type: string
                  example: "This article is very helpful"
        responses:
          200:
            description: Moderation result with sentiment, toxicity, and spam signals
            schema:
              type: object
              properties:
                decision:
                  type: string
                  enum:
                    - approve
                    - reject
                    - manual_review
                reasons:
                  type: array
                  items:
                    type: string
                sentiment:
                  type: object
                  properties:
                    label:
                      type: string
                    confidence:
                      type: number
                    scores:
                      type: object
                      additionalProperties:
                        type: number
                toxicity:
                  type: object
                  properties:
                    label:
                      type: string
                    score:
                      type: number
                    confidence:
                      type: number
                    scores:
                      type: object
                      additionalProperties:
                        type: number
                spam:
                  type: object
                  properties:
                    label:
                      type: string
                    score:
                      type: number
                    confidence:
                      type: number
                    scores:
                      type: object
                      additionalProperties:
                        type: number
                rules:
                  type: object
                  properties:
                    score:
                      type: number
                    triggers:
                      type: array
                      items:
                        type: string
          400:
            description: Missing comment field
        """
        payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        comment = payload.get("comment")
        if not comment:
            return jsonify({"error": "Field 'comment' is required"}), 400

        service.load()
        result = service.score(comment)
        return jsonify(result)
