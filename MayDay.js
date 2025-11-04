import React, { useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Polyline, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./MayDay.css";

// Leaflet marker icon fix
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

// Helpers
function toRad(v) { return (v * Math.PI) / 180; }

// Haversine distance (km) between two {lat,lng}
function haversine(a, b) {
  const R = 6371;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// Sum leg distances across a path: [pt0 -> pt1 -> ...]
function totalDistanceKm(points) {
  let sum = 0;
  for (let i = 0; i + 1 < points.length; i++) sum += haversine(points[i], points[i + 1]);
  return sum;
}

// Super simple feasibility model
function evaluateFeasibility({ distanceKm, payloadKg, batteryPct, windKts, windDirDeg }) {
  const baseRangeKm = (batteryPct / 100) * 20;      // 20km at 100%
  const payloadPenalty = payloadKg * 0.8;           // 0.8 km per kg
  const headwindFactor = 0.5 + 0.5 * Math.abs(Math.cos(toRad(windDirDeg))); // 0.5..1.0
  const windPenalty = windKts * 0.15 * headwindFactor;

  const effectiveRange = Math.max(0, baseRangeKm - payloadPenalty - windPenalty);
  const margin = effectiveRange - distanceKm;
  const feasible = margin >= 0.75;
  const confidence = Math.max(0, Math.min(1, 0.5 + margin / 10));
  return { feasible, margin, effectiveRange, confidence };
}

// Map click helper
function ClickDropper({ onClick }) {
  useMapEvents({
    click(e) {
      onClick({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
}

export default function MaydayUI() {
  const defaultCenter = { lat: 33.7455, lng: -117.8677 };

  // Mission state (JS: no generics)
  const [start, setStart] = useState(defaultCenter);
  const [waypoints, setWaypoints] = useState([]);
  const [end, setEnd] = useState(null);

  const [payloadKg, setPayloadKg] = useState(1.5);
  const [batteryPct, setBatteryPct] = useState(85);
  const [windKts, setWindKts] = useState(8);
  const [windDirDeg, setWindDirDeg] = useState(0);
  const [cruiseSpeedMS, setCruiseSpeedMS] = useState(12);

  const [useMiles, setUseMiles] = useState(false);
  const [theme, setTheme] = useState("dark");
  const [placeMode, setPlaceMode] = useState("none"); // "start" | "waypoint" | "end" | "none"
  const [allowScrollZoom, setAllowScrollZoom] = useState(false);

  const mapRef = useRef(null);

  const path = useMemo(() => {
    const pts = [];
    if (start) pts.push(start);
    for (const w of waypoints) pts.push(w);
    if (end) pts.push(end);
    return pts;
  }, [start, waypoints, end]);

  const distanceKm = useMemo(() => (path.length >= 2 ? totalDistanceKm(path) : 0), [path]);
  const distanceDisplay = useMemo(() => {
    return useMiles
      ? { value: distanceKm * 0.621371, unit: "mi" }
      : { value: distanceKm, unit: "km" };
  }, [distanceKm, useMiles]);

  const etaMinutes = useMemo(() => {
    if (cruiseSpeedMS <= 0 || distanceKm <= 0) return 0;
    const seconds = (distanceKm * 1000) / cruiseSpeedMS;
    return seconds / 60;
  }, [distanceKm, cruiseSpeedMS]);

  const [lastEval, setLastEval] = useState(null);

  const runEvaluation = () => {
    const res = evaluateFeasibility({
      distanceKm,
      payloadKg,
      batteryPct,
      windKts,
      windDirDeg,
    });
    setLastEval(res);
  };

  const resetPoints = () => {
    setStart(null);
    setWaypoints([]);
    setEnd(null);
    setLastEval(null);
  };

  const fitToRoute = () => {
    if (!mapRef.current || path.length === 0) return;
    const bounds = L.latLngBounds(path.map(p => [p.lat, p.lng]));
    mapRef.current.fitBounds(bounds.pad(0.2));
  };

  // Save / Load mission
  const saveMission = () => {
    const mission = {
      start, waypoints, end,
      payloadKg, batteryPct, windKts, windDirDeg, cruiseSpeedMS,
      useMiles,
    };
    const blob = new Blob([JSON.stringify(mission, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mayday-mission.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadInputRef = useRef(null);
  const loadMission = () => loadInputRef.current?.click();

  const onLoadFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const mission = JSON.parse(String(reader.result));
        setStart(mission.start);
        setWaypoints(mission.waypoints || []);
        setEnd(mission.end);
        setPayloadKg(mission.payloadKg ?? 1.5);
        setBatteryPct(mission.batteryPct ?? 85);
        setWindKts(mission.windKts ?? 8);
        setWindDirDeg(mission.windDirDeg ?? 0);
        setCruiseSpeedMS(mission.cruiseSpeedMS ?? 12);
        setUseMiles(!!mission.useMiles);
        setLastEval(null);
      } catch {
        alert("Invalid mission file.");
      }
    };
    reader.readAsText(file);
  };

  const handleMapClick = (p) => {
    if (placeMode === "start") setStart(p);
    else if (placeMode === "end") setEnd(p);
    else if (placeMode === "waypoint") setWaypoints(prev => [...prev, p]);
  };

  const removeWaypoint = (idx) =>
    setWaypoints(prev => prev.filter((_, i) => i !== idx));

  const swapWaypoint = (i, j) =>
    setWaypoints(prev => {
      const next = [...prev];
      if (j < 0 || j >= next.length) return next;
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });

    const bg = `mayday ${theme}`;
    const panel = "panel";
    const soft = "soft";    

  return (
    <div className={`min-h-screen w-full ${bg} flex`}>
      {/* Sidebar */}
      <aside className="w-[380px] max-w-[380px] border-r border-slate-800 p-5 space-y-5">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold tracking-tight">Mayday</h1>
          <p className={`text-sm ${soft}`}>Battery-aware routing demo (UI only)</p>
        </div>

        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Mission Inputs</h2>

          <div className="space-y-1">
            <label className="text-sm">Payload (kg): {payloadKg.toFixed(1)}</label>
            <input type="range" min={0} max={10} step={0.1}
                   value={payloadKg} onChange={(e)=>setPayloadKg(parseFloat(e.target.value))}
                   className="w-full" />
          </div>

          <div className="space-y-1">
            <label className="text-sm">Battery (%): {batteryPct}%</label>
            <input type="range" min={1} max={100} step={1}
                   value={batteryPct} onChange={(e)=>setBatteryPct(parseInt(e.target.value))}
                   className="w-full" />
          </div>

          <div className="space-y-1">
            <label className="text-sm">Wind (kts): {windKts}</label>
            <input type="range" min={0} max={40} step={1}
                   value={windKts} onChange={(e)=>setWindKts(parseInt(e.target.value))}
                   className="w-full" />
          </div>

          <div className="grid grid-cols-2 gap-2 items-center">
            <label className="text-sm">Wind From (°):</label>
            <input type="number" min={0} max={360} value={windDirDeg}
                   onChange={(e)=>setWindDirDeg(Math.max(0, Math.min(360, parseInt(e.target.value || "0"))))}
                   className="rounded-lg px-2 py-1 bg-transparent border border-slate-700" />
          </div>

          <div className="space-y-1">
            <label className="text-sm">Cruise Speed (m/s): {cruiseSpeedMS}</label>
            <input type="range" min={1} max={30} step={1}
                   value={cruiseSpeedMS} onChange={(e)=>setCruiseSpeedMS(parseInt(e.target.value))}
                   className="w-full" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <button onClick={runEvaluation}
                    disabled={path.length < 2}
                    className="rounded-2xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-400 transition px-4 py-2 text-sm font-medium">
              Evaluate
            </button>
            <button onClick={resetPoints}
                    className="rounded-2xl bg-slate-800 hover:bg-slate-700 transition px-4 py-2 text-sm font-medium">
              Reset points
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <button onClick={()=>setUseMiles(v=>!v)}
                    className="rounded-2xl bg-slate-800 hover:bg-slate-700 transition px-4 py-2 text-sm font-medium">
              Units: {useMiles ? "Miles" : "Kilometers"}
            </button>
            <button onClick={()=>setTheme(t=> t==="dark" ? "light":"dark")}
                    className="rounded-2xl bg-slate-800 hover:bg-slate-700 transition px-4 py-2 text-sm font-medium">
              Theme: {theme === "dark" ? "Dark" : "Light"}
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <button onClick={saveMission}
                    className="rounded-2xl bg-slate-800 hover:bg-slate-700 transition px-4 py-2 text-sm font-medium">
              Save Mission
            </button>
            <button onClick={loadMission}
                    className="rounded-2xl bg-slate-800 hover:bg-slate-700 transition px-4 py-2 text-sm font-medium">
              Load Mission
            </button>
            <input ref={loadInputRef} type="file" accept="application/json" className="hidden" onChange={onLoadFile}/>
          </div>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold">Status</h2>
          <div className={`rounded-2xl ${panel} border p-3 text-sm`}>
            <p>Distance: <span className="font-semibold">{distanceDisplay.value.toFixed(2)} {distanceDisplay.unit}</span></p>
            <p>ETA: <span className="font-semibold">{etaMinutes > 0 ? `${etaMinutes.toFixed(1)} min` : "--"}</span></p>
            {lastEval ? (
              <div className="mt-2 space-y-1">
                <p>Effective range: <span className="font-semibold">{lastEval.effectiveRange.toFixed(2)} km</span></p>
                <p>Margin: <span className={lastEval.margin >= 0 ? "text-emerald-500" : "text-rose-500"}>
                  {lastEval.margin.toFixed(2)} km</span></p>
                <p>Confidence: <span className="font-semibold">{Math.round(lastEval.confidence * 100)}%</span></p>
                <div className={`mt-3 rounded-xl px-3 py-2 text-center font-semibold ${
                  lastEval.feasible ? "bg-emerald-600/20 text-emerald-300 border border-emerald-800"
                                    : "bg-rose-600/20 text-rose-300 border border-rose-800"}`}>
                  {lastEval.feasible
                    ? "Conditions for delivery are ideal. Delivery simulated as successful."
                    : "Conditions for delivery are not ideal. Delivery simulated as unsuccessful."}
                </div>
              </div>
            ) : (
              <p className={soft}>Pick points and press Evaluate.</p>
            )}
          </div>
        </section>

        <section className="space-y-2">
          <h2 className="text-lg font-semibold">Waypoints</h2>
          <div className={`rounded-2xl ${panel} border p-3 text-sm space-y-2`}>
            {waypoints.length === 0 && <p className={soft}>No waypoints yet.</p>}
            {waypoints.map((w, i) => (
              <div key={i} className="flex items-center justify-between gap-2">
                <span className="truncate">{i + 1}. {w.lat.toFixed(4)}, {w.lng.toFixed(4)}</span>
                <div className="flex gap-1">
                  <button onClick={()=>swapWaypoint(i, i-1)} className="px-2 py-1 rounded border border-slate-700" title="Move up">↑</button>
                  <button onClick={()=>swapWaypoint(i, i+1)} className="px-2 py-1 rounded border border-slate-700" title="Move down">↓</button>
                  <button onClick={()=>removeWaypoint(i)} className="px-2 py-1 rounded border border-rose-700 text-rose-300" title="Remove">✕</button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </aside>

      {/* Map area */}
      <main className="flex-1 relative">
        <div className={`absolute z-10 m-4 rounded-2xl ${panel} backdrop-blur border p-3 text-sm`}>
          <p className="font-medium">Map Controls</p>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
            <button className={`rounded-xl px-3 py-2 border ${placeMode==="start" ? "border-emerald-700 bg-emerald-600/20 text-emerald-200":"border-slate-700"}`} onClick={()=>setPlaceMode("start")}>Click to set START</button>
            <button className={`rounded-xl px-3 py-2 border ${placeMode==="end" ? "border-sky-700 bg-sky-600/20 text-sky-200":"border-slate-700"}`} onClick={()=>setPlaceMode("end")}>Click to set END</button>
            <button className={`rounded-xl px-3 py-2 border ${placeMode==="waypoint" ? "border-amber-700 bg-amber-600/20 text-amber-200":"border-slate-700"}`} onClick={()=>setPlaceMode("waypoint")}>Click to add WAYPOINT</button>
            <button className="rounded-xl px-3 py-2 border border-slate-700" onClick={()=>{ setPlaceMode("none"); fitToRoute(); }}>Fit to route</button>
            <button className="rounded-xl px-3 py-2 border border-slate-700" onClick={() => setAllowScrollZoom(v => !v)}>
            {allowScrollZoom ? "Disable Scroll Zoom" : "Enable Scroll Zoom"}
            </button>
          </div>
          <p className={`mt-2 ${soft}`}>Mode: <span className="font-medium">{placeMode}</span></p>
        </div>

        <MapContainer
         center={[defaultCenter.lat, defaultCenter.lng]}
          zoom={11}
          style={{ height: "70vh", width: "100%" }}   // optional: shorter map so content stays visible
          scrollWheelZoom={false}                      // ⟵ disable wheel zoom
          touchZoom={false}                            // ⟵ optional: disable trackpad/pinch zoom
          whenCreated={(map) => { mapRef.current = map; }}
          className="z-0"
        >

          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <ClickDropper onClick={handleMapClick} />

          {start && (
            <Marker
              position={[start.lat, start.lng]}
              draggable
              eventHandlers={{
                dragend: (e) => {
                  const ll = e.target.getLatLng();
                  setStart({ lat: ll.lat, lng: ll.lng });
                },
              }}
            />
          )}

          {waypoints.map((w, i) => (
            <Marker
              key={i}
              position={[w.lat, w.lng]}
              draggable
              eventHandlers={{
                dragend: (e) => {
                  const ll = e.target.getLatLng();
                  setWaypoints(prev => {
                    const next = [...prev];
                    next[i] = { lat: ll.lat, lng: ll.lng };
                    return next;
                  });
                },
              }}
            />
          ))}

          {end && (
            <Marker
              position={[end.lat, end.lng]}
              draggable
              eventHandlers={{
                dragend: (e) => {
                  const ll = e.target.getLatLng();
                  setEnd({ lat: ll.lat, lng: ll.lng });
                },
              }}
            />
          )}

          {path.length >= 2 && (
            <Polyline positions={path.map(p => [p.lat, p.lng])} />
          )}
        </MapContainer>
      </main>
    </div>
  );
}
