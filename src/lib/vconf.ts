import { Device, type types } from "mediasoup-client";
import { io, type Socket } from "socket.io-client";

/**
 * Minimal V-Retail conference participant (the socket.io + mediasoup protocol of
 * github.com/inkincaps/conference-front, branch vretail-conference,
 * src/video-call/lib/room-service.ts and peer.ts): joins with a JWT and publishes
 * one MediaStream (audio + video) as its mic and webcam. With `receive` it also
 * gets the other participants' tracks (e.g. the backend's AI assistant's voice);
 * without, it receives nothing.
 */

/** Same as conference-front's WEBCAM_SIMULCAST_ENCODINGS. */
const WEBCAM_SIMULCAST_ENCODINGS: types.RtpEncodingParameters[] = [
  { scaleResolutionDownBy: 4, maxBitrate: 360_000 },
  { scaleResolutionDownBy: 2, maxBitrate: 640_000 },
  { scaleResolutionDownBy: 1, maxBitrate: 1_300_000 },
];

type Response = { ok: true; data: unknown } | { ok: false; errorCode: number; errorReason: string };

export type VConfCallbacks = {
  /** Connection progress or an error; null when connected and idle. */
  onStatus(status: string | null): void;
  /** Display names of the other participants, by peer id. */
  onPeers(peers: Map<string, string>): void;
  /** Any other notification from the backend (e.g. text for the agent to speak). */
  onNotification(method: string, data: unknown): void;
  /** With `receive`: a participant's audio or video track started (track) or stopped (null). */
  onRemoteTrack?(peerId: string, kind: "audio" | "video", track: MediaStreamTrack | null): void;
  /** The call ended (left, kicked out, or the connection was lost for good). */
  onClosed(reason: string): void;
};

export class VConfCall {
  private socket: Socket | null = null;
  private sendTransport: types.Transport | null = null;
  private recvTransport: types.Transport | null = null;
  private consumers = new Map<string, types.Consumer>();
  private peers = new Map<string, string>();
  private closed = false;

  constructor(
    private readonly server: string,
    private readonly jwt: string,
    /** Name shown in the call; null keeps the one in the JWT. */
    private readonly displayName: string | null,
    private readonly callbacks: VConfCallbacks,
    private readonly options: { receive?: boolean } = {}
  ) {}

  /** Connects and publishes `stream`; resolves once in the room. */
  join(stream: MediaStream): Promise<void> {
    this.callbacks.onStatus("Connecting to the call…");
    const socket = io(this.server, {
      auth: { token: this.jwt },
      transports: ["websocket"],
      path: "/v-conference",
      reconnectionAttempts: 5,
      reconnectionDelay: 2000,
    });
    this.socket = socket;
    socket.on("request", (request: { method: string; data?: any }, cb: (r: Response) => void) => // eslint-disable-line @typescript-eslint/no-explicit-any
      void this.onRequest(request, cb)
    );
    socket.on("notification", ({ method, data }: { method: string; data?: unknown }) => this.onNotification(method, data));
    socket.on("connect_error", (err) => this.callbacks.onStatus(`Can't connect to the conference server: ${err.message}`));
    socket.io.on("reconnect_failed", () => this.close("Couldn't reach the conference server."));
    socket.on("disconnect", (reason) => {
      if (reason === "io server disconnect" || reason === "io client disconnect") this.close("The conference server closed the connection.");
      else this.callbacks.onStatus("Connection lost, reconnecting…");
    });

    return new Promise((resolve, reject) => {
      socket.once("connect", () =>
        this.joinRoom(stream).then(resolve, (err) => {
          this.close(err instanceof Error ? err.message : String(err));
          reject(err);
        })
      );
    });
  }

  close(reason = "Left the call.") {
    if (this.closed) return;
    this.closed = true;
    this.socket?.removeAllListeners();
    this.socket?.io.removeAllListeners();
    this.socket?.close();
    this.sendTransport?.close();
    this.recvTransport?.close();
    for (const consumer of this.consumers.values()) {
      this.callbacks.onRemoteTrack?.(consumer.appData.peerId as string, consumer.kind, null);
    }
    this.consumers.clear();
    this.callbacks.onClosed(reason);
  }

  private request<T = any>(method: string, data?: unknown): Promise<T> { // eslint-disable-line @typescript-eslint/no-explicit-any
    return new Promise((resolve, reject) => {
      this.socket!.timeout(30_000).emit("request", { method, data }, (err: Error | null, res: Response) => {
        if (err) return reject(err);
        if (res.ok) return resolve(res.data as T);
        reject(new Error(`${method}: ${res.errorReason || `error ${res.errorCode}`}`));
      });
    });
  }

  private async joinRoom(stream: MediaStream) {
    const device = new Device();
    await device.load({ routerRtpCapabilities: await this.request("getRouterRtpCapabilities") });

    const send = await this.request("createWebRtcTransport", { forceTcp: false, producing: true, consuming: false });
    this.sendTransport = device.createSendTransport({ ...send, iceServers: [] });
    this.sendTransport.on("connect", ({ dtlsParameters }, callback, errback) => {
      this.request("connectWebRtcTransport", { transportId: this.sendTransport!.id, dtlsParameters }).then(callback, errback);
    });
    this.sendTransport.on("produce", ({ kind, rtpParameters, appData }, callback, errback) => {
      this.request<{ id: string }>("produce", { transportId: this.sendTransport!.id, kind, rtpParameters, appData }).then(
        ({ id }) => callback({ id }),
        errback
      );
    });

    if (this.options.receive) {
      const recv = await this.request("createWebRtcTransport", { forceTcp: false, producing: false, consuming: true });
      this.recvTransport = device.createRecvTransport({ ...recv, iceServers: [] });
      this.recvTransport.on("connect", ({ dtlsParameters }, callback, errback) => {
        this.request("connectWebRtcTransport", { transportId: this.recvTransport!.id, dtlsParameters }).then(callback, errback);
      });
    }

    // Without rtpCapabilities the server creates no consumers for us.
    const { peers } = await this.request<{ peers: { id: string; displayName: string }[] }>("join", {
      device: { flag: "chrome", name: "Chrome", version: navigator.userAgent.match(/Chrome\/([\d.]+)/)?.[1] ?? "" },
      rtpCapabilities: this.options.receive ? device.rtpCapabilities : undefined,
    });
    for (const peer of peers) this.peers.set(peer.id, peer.displayName);
    this.callbacks.onPeers(new Map(this.peers));
    if (this.displayName) await this.request("changeDisplayName", { displayName: this.displayName }).catch(() => {});

    const [audio] = stream.getAudioTracks();
    const [video] = stream.getVideoTracks();
    if (audio && device.canProduce("audio")) {
      await this.sendTransport.produce({ track: audio, codecOptions: { opusStereo: true, opusDtx: true } });
    }
    if (video && device.canProduce("video")) {
      await this.sendTransport.produce({
        track: video,
        encodings: WEBCAM_SIMULCAST_ENCODINGS,
        codecOptions: { videoGoogleStartBitrate: 1000 },
      });
    }
    this.callbacks.onStatus(null);
  }

  private async onRequest(request: { method: string; data?: any }, cb: (r: Response) => void) { // eslint-disable-line @typescript-eslint/no-explicit-any
    if (request.method !== "newConsumer" || !this.recvTransport) {
      return cb({ ok: false, errorCode: 403, errorReason: "I do not want to consume" });
    }
    const { peerId, producerId, id, kind, rtpParameters, appData } = request.data;
    try {
      const consumer = await this.recvTransport.consume({ id, producerId, kind, rtpParameters, appData: { ...appData, peerId } });
      this.consumers.set(consumer.id, consumer);
      consumer.on("transportclose", () => this.consumers.delete(consumer.id));
      cb({ ok: true, data: undefined }); // the server resumes the consumer once we accept
      this.callbacks.onRemoteTrack?.(peerId, consumer.kind, consumer.track);
    } catch (err) {
      cb({ ok: false, errorCode: 500, errorReason: String(err) });
    }
  }

  private onNotification(method: string, data: any) { // eslint-disable-line @typescript-eslint/no-explicit-any
    switch (method) {
      case "newPeer":
        this.peers.set(data.id, data.displayName);
        break;
      case "peerClosed":
        this.peers.delete(data.peerId);
        break;
      case "peerDisplayNameChanged":
        this.peers.set(data.peerId, data.displayName);
        break;
      case "kickedOutByModerator":
        this.close("A moderator removed the agent from the call.");
        return;
      case "consumerClosed": {
        const consumer = this.consumers.get(data.consumerId);
        if (consumer) {
          consumer.close();
          this.consumers.delete(consumer.id);
          this.callbacks.onRemoteTrack?.(consumer.appData.peerId as string, consumer.kind, null);
        }
        return;
      }
      case "samePeerOverride":
        this.close("The same agent joined from somewhere else.");
        return;
      // Media bookkeeping.
      case "producerScore":
      case "consumerScore":
      case "consumerLayersChanged":
      case "consumerPaused":
      case "consumerResumed":
      case "downlinkBwe":
      case "activeSpeaker":
        return;
      default:
        this.callbacks.onNotification(method, data);
        return;
    }
    this.callbacks.onPeers(new Map(this.peers));
  }
}
