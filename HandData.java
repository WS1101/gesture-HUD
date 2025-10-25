package org.example;
import org.json.JSONObject;
import java.util.Map;
import java.util.stream.Collectors;

public class HandData {
    public final String label;

    public final Map<String, JSONObject> landmarks;
    public HandData(String label, JSONObject landmarksJson) {
        this.label = label;
        this.landmarks = landmarksJson.keySet().stream().collect(Collectors.toMap(
                key -> key,
                key -> landmarksJson.getJSONObject(key)));
    }

    public JSONObject getLandmark(String id) {
        return landmarks.get(id);
    }


}
