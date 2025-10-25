package org.example;
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;
import java.io.IOException;
import java.net.InetSocketAddress;

public class MainServer extends WebSocketServer {
    private final GestureRouter router;

    public MainServer(int port, GestureRouter router) {
        super(new InetSocketAddress(port));
        this.router = router;
    }
    @Override
    public void onOpen(WebSocket conn, ClientHandshake handshake) {
        System.out.println("Connect!" + conn.getRemoteSocketAddress());
    }

    @Override
    public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        System.out.println("Disconnect!" + conn.getRemoteSocketAddress());
    }

    @Override
    public void onMessage(WebSocket conn, String message) {
        System.out.println("Received message: " + message);
        router.route(message);
    }

    @Override
    public void onError(WebSocket conn, Exception ex) {
        ex.printStackTrace();
    }

    @Override
    public void onStart() {
        System.out.println("Server started!(" + getPort() +")");
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        int port = 8885;

        GestureRouter gestureRouter = new GestureRouter();

        MainServer server = new MainServer(port, gestureRouter);

        server.start();


    }



}
