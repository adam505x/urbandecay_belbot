import React, { useEffect, useRef, useState } from 'react';
import './LandingPage.css';

interface LandingPageProps {
  onTryItOut: () => void;
}

interface TimelineSection {
  id: string;
  title: string;
  content: string;
  image: string;
}

const timelineSections: TimelineSection[] = [
  {
    id: 'satellite-intelligence',
    title: 'Sentinel Satellite Intelligence',
    content: 'Belfast Sentinel ingests Sentinel-2 vegetation/built-up indices and Sentinel-5P NO₂ to detect early signals of urban decay — loss of greenery, rising surface heat, traffic-pollution corridors — at 20 m resolution across every 500 m grid cell in the Belfast metro.',
    image: 'https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=1200&auto=format&fit=crop&q=60'
  },
  {
    id: 'ni-open-data',
    title: 'Northern Ireland Open Data',
    content: 'Layered with DfI Strategic Flood Maps (river, coastal, surface water, climate-projected), NIMDM 2017 multiple-deprivation deciles, NISRA Census 2021 dwellings & tenure, and the NI House Price Index by LGD — every public signal that hints at decline is fused into one model.',
    image: 'https://images.unsplash.com/photo-1584291527935-456e8e2dd734?q=80&w=2050&auto=format&fit=crop'
  },
  {
    id: 'decision-support',
    title: 'Belfast City Council Decision Support',
    content: 'A LightGBM classifier turns those signals into a probability per grid cell, served through a FastAPI backend. Council officers can target funding, intervene before decay sets in, and route at-risk residents to housing — all from an interactive map.',
    image: 'https://images.unsplash.com/photo-1605810230434-7631ac76ec81?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
  }
];

const LandingPage: React.FC<LandingPageProps> = ({ onTryItOut }) => {
  const [activeSection, setActiveSection] = useState(0);
  const [showTimeline, setShowTimeline] = useState(false);
  const sectionsRef = useRef<(HTMLDivElement | null)[]>([]);
  const heroRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const observerOptions = {
      root: null,
      rootMargin: '-20% 0px -20% 0px',
      threshold: 0.5
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const sectionIndex = sectionsRef.current.findIndex(
            (ref) => ref === entry.target
          );
          if (sectionIndex !== -1) {
            setActiveSection(sectionIndex);
          }
        }
      });
    }, observerOptions);

    // Observer for hero section to control timeline visibility
    const heroObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.target === heroRef.current) {
          setShowTimeline(!entry.isIntersecting);
        }
      });
    }, {
      root: null,
      rootMargin: '0px',
      threshold: 0.1
    });

    const currentSections = sectionsRef.current;
    const currentHero = heroRef.current;

    currentSections.forEach((section) => {
      if (section) observer.observe(section);
    });

    if (currentHero) {
      heroObserver.observe(currentHero);
    }

    return () => {
      currentSections.forEach((section) => {
        if (section) observer.unobserve(section);
      });
      if (currentHero) {
        heroObserver.unobserve(currentHero);
      }
    };
  }, []);

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <div className="hero-section" ref={heroRef}>
        <header className="landing-header">
          <h1 className="brand-title">BELFAST SENTINEL</h1>
        </header>
        
        <main className="landing-main">
          <div className="content-section">
            {/* Subtitle overlays that will be behind the image */}
            <div className="background-text">
              <div className="subtitle-overlay subtitle-overlay-1">
                <span className="subtitle-text">Predict • Prevent • Protect</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-2">
                <span className="subtitle-text1">Sentinel-2 + Sentinel-5P</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-3">
                <span className="subtitle-text2">NIMDM &amp; DfI Open Data</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-4">
                <span className="subtitle-text3">For Belfast City Council</span>
              </div>
            </div>
            
            <div className="image-section">
              <div className="toronto-image">
                <img
                  src="https://images.unsplash.com/photo-1564591926582-fd8aa2b58fc4?auto=format&fit=crop&w=1000&q=80"
                  alt="Belfast skyline — urban intelligence"
                  className="skyline-img"
                />
              </div>
            </div>
            
            <div className="text-section">
              <h1 className="main-title">
                Preventing Urban Decay in Belfast
                <span className="subtitle">Before It Happens</span>
              </h1>

              <div className="description">
                <p className="lead-text">
                  <strong>Urban decay costs Northern Ireland millions every year</strong>
                  in lost property value, displaced residents, and reactive remediation.
                  Belfast Sentinel flips the model — using Sentinel satellite imagery,
                  NIMDM deprivation data and DfI flood maps to flag at-risk
                  neighbourhoods before the decline is visible from the street.
                </p>

                <div className="stats-highlight">
                  <div className="stat">
                    <span className="stat-number">2,496</span>
                    <span className="stat-label">Belfast Grid Cells</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">36</span>
                    <span className="stat-label">Predictive Features</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">500 m</span>
                    <span className="stat-label">Spatial Resolution</span>
                  </div>
                </div>
                
                <button className="cta-button" onClick={onTryItOut}>
                  <span>Explore Live Demo</span>
                  <div className="button-glow"></div>
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Fixed Timeline Sidebar - Only show when not on hero section */}
      <div className={`timeline-sidebar ${showTimeline ? 'visible' : 'hidden'}`}>
        <div className="timeline-line"></div>
        <div 
          className="timeline-indicator"
          style={{
            transform: `translateY(${activeSection * 33.33 + 6.8}vh)`
          }}
        ></div>
        
        <div className="timeline-labels">
          {timelineSections.map((section, index) => (
            <div 
              key={section.id}
              className={`timeline-label ${activeSection === index ? 'active' : ''}`}
              style={{ top: `${index * 33.33 + 11}vh` }}
            >
              <div className="timeline-dot"></div>
              <h3>{section.title}</h3>
            </div>
          ))}
        </div>
      </div>

      {/* Full-Page Timeline Sections */}
      {timelineSections.map((section, index) => (
        <div
          key={section.id}
          ref={(el) => {
            sectionsRef.current[index] = el;
          }}
          className={`timeline-page ${activeSection === index ? 'active' : ''}`}
          data-section={index}
        >
          <div className={`timeline-page-content ${showTimeline ? 'with-sidebar' : 'without-sidebar'}`}>
            <div className="section-image">
              <img src={section.image} alt={section.title} />
              <div className="image-overlay"></div>
            </div>
            <div className="section-text">
              <h2>{section.title}</h2>
              <p>{section.content}</p>
              
              {index === 0 && (
                <div className="tech-specs">
                  <div className="tech-item">🛰️ Sentinel-2 NDVI / NDBI / NDWI</div>
                  <div className="tech-item">🌫️ Sentinel-5P tropospheric NO₂</div>
                  <div className="tech-item">🌡️ Surface temperature trends</div>
                </div>
              )}

              {index === 1 && (
                <div className="tech-specs">
                  <div className="tech-item">🌊 DfI Strategic Flood Maps</div>
                  <div className="tech-item">📊 NIMDM 2017 deprivation deciles</div>
                  <div className="tech-item">🏘️ NISRA Census 2021 + NI HPI</div>
                </div>
              )}

              {index === 2 && (
                <div className="tech-specs">
                  <div className="tech-item">⚡ LightGBM gradient boosting</div>
                  <div className="tech-item">🗺️ Mapbox GL interactive grid</div>
                  <div className="tech-item">🏛️ Funding-allocation ready</div>
                </div>
              )}
            </div>
          </div>
        </div>
              ))}
    </div>
  );
};

export default LandingPage; 