from model import NeuralNetwork, ModelActions
from hand_camera import FingerDrawer

input_size = 784
num_classes = 10

net = NeuralNetwork(input_size, num_classes)
net_actions = ModelActions(net)
net_actions.load_model()
net_actions.save_model()
drawer = FingerDrawer(net_actions)
drawer.hand_draw()