package org.example;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class LandmarkHandler implements GestureHandler {
    private final GestureListener listener;
    public LandmarkHandler(GestureListener listener) {
        this.listener = listener;
    }

    @Override
    public void handle(JSONObject data) {
        System.out.println("Received data: " + data);
        if(listener == null) return;

        try{
            if(data.has("hands")) {
                JSONArray allhands = data.getJSONArray("hands");
                System.out.println(allhands);
                for(int i = 0; i < allhands.length(); i++) {
                    JSONObject handJson = allhands.getJSONObject(i);
                    String handlabel = handJson.getString("hand_label");
                    JSONObject landmarks = handJson.getJSONObject("landmarks");

                    System.out.println("---" + handlabel + "---");

                    for(String landmarkId : landmarks.keySet()) {
                        JSONObject coords = landmarks.getJSONObject(landmarkId);
                        double x = coords.getDouble("x");
                        double y = coords.getDouble("y");
                        System.out.println(String.format("Landmark %s: (x = %.3f, y = %.3f)", handlabel, x, y));
                    }
                }
                if(allhands.length() > 0){
                    JSONObject firstHand = allhands.getJSONObject(0);
                    if(firstHand.has("landmarks")) {
                        JSONObject landmarks = firstHand.getJSONObject("landmarks");
                        if(landmarks.has("8")) {
                            JSONObject indexTip = landmarks.getJSONObject("8");
                            listener.onHandMoved(indexTip.getDouble("x"), indexTip.getDouble("y"));
                        }
                    }
                }
            }
            else{
                System.out.println("No hands found");
            }
        }catch(JSONException e){
            System.err.println("JSON Exception: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
