// GameTreeNode.jsx
import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';

export const GameTreeNode = memo(({ data }) => {
  const content = data.renderFn ? data.renderFn(data.seat) : <span>?</span>;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minWidth: '20px',
      minHeight: '20px',
      position: 'relative' // Ensure handles are positioned relative to this box
    }}>
      {/* TARGET: Where lines enter (Left side) */}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#555', width: '6px', height: '6px' }}
      />

      {content}

      {/* SOURCE: Where lines leave (Right side) */}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#555', width: '6px', height: '6px' }}
      />
    </div>
  );
});
