import React, { useMemo } from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import { getLayoutElements } from './GameTreeUtils';
import { GameTreeNode } from './GameTreeNode';

const nodeTypes = {
  gameNode: GameTreeNode,
};

export function GameTreePanel({ history, seatToHtml }) {

  const { nodes, edges } = useMemo(() => {
    return getLayoutElements(history, seatToHtml);
  }, [history, seatToHtml]);

  return (
    <div style={{ height: '300px', border: '1px solid #444', marginTop: '10px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable={false}
      >
        <Background color="#555" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
