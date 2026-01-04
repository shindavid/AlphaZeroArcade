import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';

export const GameTreeNode = memo(({ data }) => {
  // data.renderFn is your passed-down seatToHtml function
  // data.seat is the 'B' or 'W' value
  const content = data.renderFn ? data.renderFn(data.seat) : <span>?</span>;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      // Optional: Add a subtle border or background if the span is transparent
      minWidth: '20px',
      minHeight: '20px'
    }}>
      {/* Input Handle (Top) */}
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: '#555', width: '6px', height: '6px' }}
      />

      {/* Your Existing Visualization */}
      {content}

      {/* Output Handle (Bottom) - Only show if we expect children, or always show */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: '#555', width: '6px', height: '6px' }}
      />
    </div>
  );
});
