import React, { useState } from 'react';
import { FiHelpCircle, FiRotateCw } from 'react-icons/fi';

const FlipCard = ({ 
  icon: Icon, 
  value, 
  label, 
  bgColor, 
  borderColor, 
  iconColor, 
  valueColor,
  explanation,
  example
}) => {
  const [isFlipped, setIsFlipped] = useState(false);

  const handleFlip = () => {
    setIsFlipped(!isFlipped);
  };

  return (
    <div 
      style={{ 
        perspective: '1000px',
        height: '120px',
        cursor: 'pointer'
      }}
      onClick={handleFlip}
    >
      <div style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        transformStyle: 'preserve-3d',
        transition: 'transform 0.6s',
        transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
      }}>
        {/* Front Side */}
        <div style={{ 
          position: 'absolute',
          width: '100%',
          height: '100%',
          backfaceVisibility: 'hidden',
          background: bgColor, 
          border: `1px solid ${borderColor}`,
          borderRadius: '10px', 
          padding: '1rem',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}>
          <div style={{ position: 'absolute', top: '6px', right: '6px' }}>
            <FiHelpCircle style={{ color: '#64748b', fontSize: '0.9rem', opacity: 0.6 }} />
          </div>
          <Icon style={{ color: iconColor, fontSize: '1.25rem', marginBottom: '0.5rem' }} />
          <div style={{ fontSize: '1.5rem', fontWeight: '700', color: valueColor }}>
            {value}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{label}</div>
        </div>

        {/* Back Side */}
        <div style={{ 
          position: 'absolute',
          width: '100%',
          height: '100%',
          backfaceVisibility: 'hidden',
          transform: 'rotateY(180deg)',
          background: 'rgba(30, 41, 59, 0.95)', 
          border: `1px solid ${borderColor}`,
          borderRadius: '10px', 
          padding: '0.75rem',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          overflow: 'hidden'
        }}>
          <div style={{ position: 'absolute', top: '6px', right: '6px' }}>
            <FiRotateCw style={{ color: '#64748b', fontSize: '0.8rem', opacity: 0.6 }} />
          </div>
          <div style={{ 
            fontSize: '0.7rem', 
            fontWeight: '600', 
            color: valueColor, 
            marginBottom: '0.35rem',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}>
            {label}
          </div>
          <div style={{ 
            fontSize: '0.68rem', 
            color: '#e2e8f0', 
            lineHeight: '1.4',
            marginBottom: '0.35rem'
          }}>
            {explanation}
          </div>
          {example && (
            <div style={{ 
              fontSize: '0.62rem', 
              color: '#94a3b8', 
              fontStyle: 'italic',
              borderTop: '1px solid rgba(148, 163, 184, 0.2)',
              paddingTop: '0.35rem',
              marginTop: 'auto'
            }}>
              {example}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FlipCard;
