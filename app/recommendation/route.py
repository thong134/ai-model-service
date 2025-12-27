from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .destination import DestinationRecommender
from .optimizer import build_graph, optimize_route

@dataclass
class RouteRecommender:
    dest_recommender: DestinationRecommender

    def recommend_route(
        self,
        user_hobbies: List[str],
        user_favorites: List[str],
        province: str,
        start_date: str,  # Format: "dd/MM/yyyy"
        end_date: str,    # Format: "dd/MM/yyyy"
        start_location: Optional[Dict[str, float]] = None,
        max_spots_per_day: int = 5
    ) -> Dict[str, object]:
        
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%d/%m/%Y")
            end_dt = datetime.strptime(end_date, "%d/%m/%Y")
        except ValueError:
            return {"error": "Invalid date format. Use dd/MM/yyyy"}
        
        num_days = (end_dt - start_dt).days + 1
        if num_days <= 0:
            return {"error": "endDate must be after startDate"}
        
        # 1. Select Candidates
        needed_spots = num_days * max_spots_per_day * 2
        candidates = self.dest_recommender.recommend(
            user_hobbies=user_hobbies,
            user_favorites=user_favorites,
            province=province,
            top_n=max(20, needed_spots)
        )
        
        if not candidates:
            return {"error": "No suitable destinations found in this province"}

        candidate_ids = [c['destinationId'] for c in candidates]
        
        # Get full details
        dest_df = self.dest_recommender.destinations
        selected_dests = dest_df[dest_df['destinationId'].astype(str).isin(candidate_ids)].copy()
        
        # Prepare waypoints for optimizer
        waypoints_data = []
        for _, row in selected_dests.iterrows():
            waypoints_data.append({
                "id": str(row['destinationId']),
                "name": row['name'],
                "coords": (row['latitude'], row['longitude']),
                "label": row['name']
            })

        target_count = min(len(waypoints_data), num_days * max_spots_per_day)
        final_waypoints = waypoints_data[:target_count]
        
        if not final_waypoints:
            return {"error": "Not enough destinations to create a route"}
        
        # 2. Optimize Route Order
        start_node = final_waypoints[0]  # Start from first recommended spot
        if start_location:
            start_node = {
                "id": "start_point",
                "coords": (start_location['latitude'], start_location['longitude']),
                "label": "Start Location"
            }
        
        graph = build_graph(final_waypoints, start_point=start_node if start_location else None)
        
        solution = optimize_route(
            graph=graph,
            start=start_node['id'],
            generations=100,
            population_size=50
        )
        
        # 3. Build Stops with dayOrder, sequence, times
        stops = []
        sequence_in_day = 0
        current_day = 1
        spots_today = 0
        
        # Time allocation: assume each destination visit is ~2 hours, start at 8:00
        daily_start_hour = 8
        visit_duration_hours = 2
        travel_buffer_hours = 0.5  # 30 min between spots
        
        for idx, node_id in enumerate(solution.route):
            if node_id == "start_point":
                continue  # Skip the user's start location
            
            # Check if we need to move to next day
            if spots_today >= max_spots_per_day:
                current_day += 1
                spots_today = 0
                sequence_in_day = 0
                
            if current_day > num_days:
                break  # No more days available
            
            # Calculate time for this stop
            start_hour = daily_start_hour + spots_today * (visit_duration_hours + travel_buffer_hours)
            end_hour = start_hour + visit_duration_hours
            
            # Format times as HH:mm
            start_time = f"{int(start_hour):02d}:{int((start_hour % 1) * 60):02d}"
            end_time = f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}"
            
            sequence_in_day += 1
            spots_today += 1
            
            # Get destination info
            dest_info = selected_dests[selected_dests['destinationId'].astype(str) == node_id]
            dest_name = dest_info.iloc[0]['name'] if not dest_info.empty else "Unknown"
            
            stops.append({
                "dayOrder": current_day,
                "sequence": sequence_in_day,
                "destinationId": int(node_id) if node_id.isdigit() else node_id,
                "startTime": start_time,
                "endTime": end_time,
                "notes": f"Đề xuất tự động: {dest_name}"
            })

        return {
            "name": f"Lộ trình {province} {num_days} ngày",
            "province": province,
            "startDate": start_date,
            "endDate": end_date,
            "stops": stops
        }
