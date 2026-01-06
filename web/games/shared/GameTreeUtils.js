import { MarkerType, Position } from 'reactflow';

const NODE_WIDTH = 40;
const NODE_HEIGHT = 40;
const MOVE_SPACING = 60;
const LANE_SPACING = 50;

export function getLayoutElements(history, renderFn) {
  if (!history || history.length === 0) return { nodes: [], edges: [] };

  const treeMap = buildTreeStructure(history);
  const { nodes, edges } = calculateCoordinates(treeMap, renderFn);
  return { nodes, edges };
}

function buildTreeStructure(history) {
  const nodeMap = new Map();
  nodeMap.set("0", {
    id: "0",
    children: [],
    data: { label: "Start", seat: null, moveNumber: 0, index: 0 }
  });

  const sortedHistory = Array.from(history.values()).sort((a, b) => a.index - b.index);

  sortedHistory.forEach((msg) => {
    const id = String(msg.index);
    const parentId = String(msg.parent_index);
    if (!nodeMap.has(parentId)) return;

    const parent = nodeMap.get(parentId);
    const newNode = {
      id,
      parentId,
      children: [],
      data: { ...msg, moveNumber: parent.data.moveNumber + 1 }
    };

    nodeMap.set(id, newNode);
    parent.children.push(id);
  });
  return nodeMap;
}

function calculateCoordinates(nodeMap, renderFn) {
  const ctx = {
    nodes: [],
    edges: [],
    nextLaneAtMove: new Map(),
    nodeMap: nodeMap,
    renderFn: renderFn
  };

  traverse("0", 0, ctx);
  return { nodes: ctx.nodes, edges: ctx.edges };
}

function traverse(nodeId, inheritedY, ctx) {
  const node = ctx.nodeMap.get(nodeId);
  if (!node) return;

  const x = node.data.moveNumber * MOVE_SPACING;
  let y = inheritedY;

  const nextAvailableLane = ctx.nextLaneAtMove.get(node.data.moveNumber) || 0;

  if (y === -1 || y < nextAvailableLane) {
     y = nextAvailableLane;
  }

  ctx.nextLaneAtMove.set(node.data.moveNumber, y + LANE_SPACING);


  ctx.nodes.push({
    id: node.id,
    position: { x, y },
    type: 'gameNode',
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    data: {
      ...node.data,
      renderFn: ctx.renderFn
    },
    style: { width: NODE_WIDTH, height: NODE_HEIGHT }
  });

  // --- Add Edge ---
  if (node.parentId) {
    ctx.edges.push({
      id: `e${node.parentId}-${node.id}`,
      source: node.parentId,
      target: node.id,
      type: 'default',
      markerEnd: { type: MarkerType.ArrowClosed },
      style: { stroke: '#b1b1b7' }
    });
  }

  if (node.children.length > 0) {
    traverse(node.children[0], y, ctx);

    for (let i = 1; i < node.children.length; i++) {
      traverse(node.children[i], -1, ctx);
    }
  }
}
