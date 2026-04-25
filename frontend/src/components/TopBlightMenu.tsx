import React, { useState } from 'react';
import { RiskGridCell } from '../services/api';
import './TopBlightMenu.css';

interface TopBlightMenuProps {
  riskData: RiskGridCell[] | null;
}

interface BlightTypeCount {
  type: string;
  count: number;
  percentage: number;
}

const TopBlightMenu: React.FC<TopBlightMenuProps> = ({ riskData }) => {
  const [isOpen, setIsOpen] = useState(false);
  // Calculate top blight types from the data
  const getTopBlightTypes = (): BlightTypeCount[] => {
    if (!riskData || riskData.length === 0) return [];

    const blightCounts: { [key: string]: number } = {};
    let totalWithBlight = 0;

    // Count occurrences of each blight type
    riskData.forEach(cell => {
      // Use overall_most_common_blight for analysis
      const blightType = cell.overall_most_common_blight;
      
      if (blightType && blightType !== 'None' && blightType !== 'null' && blightType.trim() !== '') {
        blightCounts[blightType] = (blightCounts[blightType] || 0) + 1;
        totalWithBlight++;
      }
    });

    // Convert to array and sort by count
    const sortedBlights = Object.entries(blightCounts)
      .map(([type, count]) => ({
        type: formatBlightType(type),
        count,
        percentage: Math.round((count / totalWithBlight) * 100)
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5); // Top 5

    return sortedBlights;
  };

  // Format blight type names for display
  const formatBlightType = (type: string): string => {
    if (!type) return 'Unknown';
    
    // Map NI decay-driver codes to short, readable labels
    const nameMap: { [key: string]: string } = {
      'unoccupied_dwellings': 'Unoccupied Dwellings',
      'surface_flooding': 'Surface-Water Flooding',
      'low_vegetation': 'Loss of Vegetation',
      'high_no2': 'High Air Pollution (NO₂)',
      'social_housing_concentration': 'Social-Housing Pressure',
      'crime_pressure': 'Crime Pressure',
      'flood_river': 'River Flood Risk',
      'flood_coastal': 'Coastal Flood Risk',
      'derelict_buildings': 'Derelict Buildings',
      'low_house_price_growth': 'Stagnant House Prices',
    };
    
    // Use mapped name if available, otherwise clean and format
    if (nameMap[type]) {
      return nameMap[type];
    }
    
    return type
      .replace(/_/g, ' ')
      .replace(/\//g, ' / ')
      .toLowerCase()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
  };



  const topBlights = getTopBlightTypes();
  
  // Get total areas and areas with blight data for context
  const totalAreas = riskData?.length || 0;
  const areasWithBlightData = riskData?.filter(cell => 
    cell.overall_most_common_blight && 
    cell.overall_most_common_blight !== 'None' && 
    cell.overall_most_common_blight !== 'null' && 
    cell.overall_most_common_blight.trim() !== ''
  ).length || 0;

  if (!riskData || topBlights.length === 0) {
    return null;
  }

  return (
    <div className={`top-blight-menu ${isOpen ? 'open' : 'closed'}`}>
      <div className="blight-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="header-content">
          <h3> Top Decay Drivers</h3>
          <span className="blight-subtitle">{totalAreas.toLocaleString()} Belfast Grid Cells</span>
        </div>
        <div className="toggle-arrow">
          {isOpen ? '▲' : '▼'}
        </div>
      </div>
      
      {isOpen && (
        <>
          <div className="blight-list">
            {topBlights.map((blight, index) => (
              <div key={blight.type} className="blight-item">
                <div className="blight-indicator">
                  <span className="blight-rank">#{index + 1}</span>
                </div>
                
                <div className="blight-info">
                  <div className="blight-name">{blight.type}</div>
                  <div className="blight-stats">
                    <span className="blight-count">{blight.count.toLocaleString()} areas</span>
                    <span className="blight-percentage">{blight.percentage}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="blight-footer">
            <small>Dominant driver per cell from the LightGBM model<br/>
            {areasWithBlightData.toLocaleString()} cells with an identified driver out of {totalAreas.toLocaleString()} total</small>
          </div>
        </>
      )}
    </div>
  );
};

export default TopBlightMenu; 