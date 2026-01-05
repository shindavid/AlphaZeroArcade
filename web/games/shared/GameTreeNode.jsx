import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import './GameTreeNode.css';

export const GameTreeNode = memo(({ data }) => {
  const content = data.renderFn ? data.renderFn(data.seat) : <span>?</span>;

  return (
    <div
      className="game-node-wrapper"
      move-number={data.moveNumber}
    >
      {data.label !== "Start" && (
        <Handle
          type="target"
          position={Position.Left}
          className="game-handle"
        />
      )}

      {content}

      {/* SOURCE: Where lines leave (Right side) */}
      <Handle
        type="source"
        position={Position.Right}
        className="game-handle"
      />
    </div>
  );
});
