import React from 'react';
import { LeafletCityMap } from '@/components/common/LeafletCityMap';

export const StylizedCityMap = ({ points = [], onSelect, onMapClick, pickerHint, selectedId, showHotspots = true, className, height = 460 }) => (
  <LeafletCityMap
    points={points}
    onSelect={onSelect}
    onMapClick={onMapClick}
    pickerHint={pickerHint}
    selectedId={selectedId}
    showHotspots={showHotspots}
    className={className}
    height={height}
  />
);
