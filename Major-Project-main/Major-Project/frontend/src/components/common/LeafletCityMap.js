import React, { useEffect, useMemo } from 'react';
import { Activity, MapPinned } from 'lucide-react';
import { Circle, CircleMarker, MapContainer, Popup, TileLayer, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { PRIORITY_META } from '@/lib/constants';
import { cn } from '@/lib/utils';

const BENGALURU = [12.9716, 77.5946];
const PRIORITY_COLORS = {
  LOW: '#3182bd', MEDIUM: '#e69500', HIGH: '#f06418', CRITICAL: '#d9272e',
};

const MapViewport = ({ points }) => {
  const map = useMap();
  useEffect(() => {
    if (points.length === 1) {
      map.setView([Number(points[0].location.lat), Number(points[0].location.lng)], 14, { animate: true });
    } else if (points.length > 1) {
      map.fitBounds(points.map((point) => [Number(point.location.lat), Number(point.location.lng)]), { padding: [36, 36], maxZoom: 13 });
    } else {
      map.setView(BENGALURU, 11);
    }
  }, [map, points]);
  return null;
};

const MapClickHandler = ({ onMapClick }) => {
  useMapEvents({
    click: (event) => onMapClick?.({ lat: event.latlng.lat, lng: event.latlng.lng }),
  });
  return null;
};

export const LeafletCityMap = ({ points = [], onSelect, onMapClick, pickerHint, selectedId, showHotspots = true, className, height = 460 }) => {
  const validPoints = useMemo(() => points.filter((point) => (
    Number.isFinite(Number(point?.location?.lat)) && Number.isFinite(Number(point?.location?.lng))
  )), [points]);
  const hotspots = useMemo(() => {
    const wards = new Map();
    validPoints.forEach((point) => {
      const ward = point.location?.ward || 'Bengaluru';
      const current = wards.get(ward) || { lat: 0, lng: 0, count: 0 };
      current.lat += Number(point.location.lat);
      current.lng += Number(point.location.lng);
      current.count += 1;
      wards.set(ward, current);
    });
    if (!showHotspots) return [];
    return [...wards.entries()]
      .filter(([, ward]) => ward.count >= 3)
      .map(([ward, value]) => ({
        ward,
        center: [value.lat / value.count, value.lng / value.count],
        count: value.count,
      }));
  }, [validPoints, showHotspots]);

  return (
    <div className={cn('relative overflow-hidden rounded-2xl border border-border bg-muted shadow-card', className)}>
      <div className="absolute inset-x-0 top-0 z-[500] flex items-center justify-between border-b border-white/70 bg-white/85 px-4 py-3 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary/10 text-primary"><MapPinned className="h-4 w-4" /></span>
          <div><div className="text-sm font-semibold text-foreground">Bengaluru issue map</div><div className="text-[11px] text-muted-foreground">Live civic reports and priority hotspots</div></div>
        </div>
        <div className="inline-flex items-center gap-1.5 rounded-full border border-success/20 bg-success/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-success"><Activity className="h-3 w-3" /> {validPoints.length} active</div>
      </div>
      <MapContainer center={BENGALURU} zoom={11} scrollWheelZoom={height > 240} zoomControl={height > 240}
        className={onMapClick ? 'cursor-crosshair' : undefined} style={{ height, width: '100%' }}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapViewport points={validPoints} />
        {onMapClick && <MapClickHandler onMapClick={onMapClick} />}
        {hotspots.map((hotspot) => (
          <Circle
            key={hotspot.ward}
            center={hotspot.center}
            radius={350 + hotspot.count * 45}
            pathOptions={{ color: '#f06418', opacity: 0, fillColor: '#f06418', fillOpacity: 0.12 }}
          />
        ))}
        {validPoints.map((point) => {
          const selected = selectedId === point.id;
          const color = PRIORITY_COLORS[point.priority] || PRIORITY_COLORS.LOW;
          return (
            <CircleMarker
              key={point.id}
              center={[Number(point.location.lat), Number(point.location.lng)]}
              radius={selected ? 11 : point.priority === 'CRITICAL' ? 9 : 7}
              pathOptions={{ color: '#fff', weight: selected ? 3 : 2, fillColor: color, fillOpacity: selected ? 1 : 0.88 }}
              eventHandlers={{ click: () => onSelect?.(point) }}
            >
              <Popup>
                <div className="min-w-[190px] text-xs">
                  <div className="font-semibold text-slate-900">{point.title || 'Civic issue'}</div>
                  <div className="mt-1 text-slate-500">{point.location?.ward || 'Bengaluru'} · {PRIORITY_META[point.priority]?.label || point.priority || 'Low'}</div>
                  <button type="button" className="mt-2 font-semibold text-blue-700" onClick={() => onSelect?.(point)}>View report</button>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
      {pickerHint && (
        <div className="pointer-events-none absolute left-3 top-[4.25rem] z-[500] rounded-lg border border-border bg-card/90 px-2.5 py-1.5 text-[11px] font-medium text-foreground shadow-sm backdrop-blur">
          {pickerHint}
        </div>
      )}
      <div className="absolute bottom-3 left-3 z-[500] flex flex-wrap gap-x-3 gap-y-1 rounded-xl border border-border bg-card/90 px-3 py-2 text-[11px] font-medium backdrop-blur">
        {Object.keys(PRIORITY_META).map((priority) => <span key={priority} className="inline-flex items-center gap-1.5 text-muted-foreground"><span className="h-2.5 w-2.5 rounded-full" style={{ background: PRIORITY_COLORS[priority] }} />{PRIORITY_META[priority].label}</span>)}
      </div>
      <a
        href="https://www.openstreetmap.org/copyright"
        target="_blank"
        rel="noreferrer"
        className={cn(
          'absolute right-3 z-[500] rounded-lg border border-border bg-card/90 px-2.5 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground backdrop-blur',
          height <= 240 ? 'bottom-12' : 'bottom-3',
        )}
      >
        Bengaluru · live view · © OpenStreetMap contributors
      </a>
    </div>
  );
};
