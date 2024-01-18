import os
import torch
import random
from matplotlib import pyplot as plt
import numpy as np

class Tester:
    def __init__(self, project_name, model, device, test_data_loader, test_datasets, transforms, checkpoint_file_path):
        self.project_name = project_name
        self.model = model
        self.device = device
        self.test_data_loader = test_data_loader
        self.test_datasets = test_datasets
        self.transforms = transforms
        self.lastest_file_path = os.path.join(checkpoint_file_path, f'{project_name}_cp_lastest.pt')

        self.model = model.load_state_dict(torch.load(self.lastest_file_path, map_location=torch.device(device)))
        print(f"Model Path: {self.lastest_file_path}")

        
    def do_test(self):
        self.model.eval()

        num_corrects_test = 0
        num_tested_samples = 0

        with torch.no_grad():
            for test_batch in self.test_data_loader:
                input_test, target_test = test_batch

                input_test = self.transforms(input_test)

                output_test, _, _ = self.model(input_test)

                predicted_test = torch.argmax(output_test, dim=1)
                num_corrects_test += torch.sum(torch.eq(predicted_test, target_test)).item()
                num_tested_samples += len(input_test)

        test_accuracy = 100 * num_corrects_test / num_tested_samples
        print(f'Test Results: {test_accuracy:6.3f}')


    def test_random_input(self):
        self.model.eval()
        
        rand = random.randint(0, 10000)
        with torch.no_grad():
            input, target = self.test_datasets[rand]
            input = torch.tensor(np.array(input)).permute(2, 0, 1).unsqueeze(dim=0)
            input = self.transforms(input)
            output = self.model(input)

            predicted = torch.argmax(output, dim=1).item()
            
            plt.show(input)
            print(f"Model predicted: {predicted}, Label: {target}")

            if predicted == target:
                print("Correct answer")
            else:
                print("Wrong answer")



