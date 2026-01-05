import React, { useMemo } from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import './GameTree.css'

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
    <div className="game-tree-container">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable={false}
      >
        <Background className="game-tree-background" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
