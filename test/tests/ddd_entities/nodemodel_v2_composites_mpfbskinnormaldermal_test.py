import bpy, os
from pytest import approx
from .. import dynamic_import
from .. import ObjectService
from .. import NodeService
NodeWrapperMpfbSkinNormalDermal = dynamic_import("mpfb.entities.nodemodel.v2.composites.nodewrappermpfbskinnormaldermal", "NodeWrapperMpfbSkinNormalDermal")

def test_composite_is_available():
    assert NodeWrapperMpfbSkinNormalDermal

def test_composite_can_create_instance():
    node_tree_name = ObjectService.random_name()
    node_tree = NodeService.create_node_tree(node_tree_name)
    node = NodeWrapperMpfbSkinNormalDermal.create_instance(node_tree)
    assert node
    assert node.node_tree.name == "MpfbSkinNormalDermal"
    assert "Group Output" in node.node_tree.nodes
    assert "Math" in node.node_tree.nodes
    assert "Voronoi Texture" in node.node_tree.nodes
    assert "Group.001" in node.node_tree.nodes
    assert "Group" in node.node_tree.nodes
    assert "Texture Coordinate" in node.node_tree.nodes
    assert "Bump" in node.node_tree.nodes
    assert "Group Input" in node.node_tree.nodes
    has_link_to_output = False
    for link in node.node_tree.links:
        if link.to_node.name == "Group Output":
            has_link_to_output = True
    assert has_link_to_output
    node_tree.nodes.remove(node)
    NodeService.destroy_node_tree(node_tree)

def test_composite_validate_tree():
    node_tree_name = ObjectService.random_name()
    node_tree = NodeService.create_node_tree(node_tree_name)
    node = NodeWrapperMpfbSkinNormalDermal.create_instance(node_tree)
    assert NodeWrapperMpfbSkinNormalDermal.validate_tree_against_original_def()
    node_tree.nodes.remove(node)
    NodeService.destroy_node_tree(node_tree)
