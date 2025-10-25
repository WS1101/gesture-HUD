package org.example;
import org.json.JSONObject;
import java.util.HashMap;
import java.util.Map;

public class GestureRouter {
    final Map<String, GestureHandler> handlerMap = new HashMap<>();
    public GestureRouter() {
    }
    public void registerHandler(String gestureKey, GestureHandler handler) {
        this.handlerMap.put(gestureKey, handler);
    }
    public void route(String jsonMessage) {
        try {
            JSONObject messageJson = new JSONObject(jsonMessage);


            if(messageJson.has("hands")) {
                GestureHandler handler = handlerMap.get("HAND_LANDMARKS");


                if(handler != null) {
                    handler.handle(messageJson);

                }
            }
        }
        catch(Exception e) {
            System.err.println("JSON Error: " + e.getMessage());
        }

    }
}
