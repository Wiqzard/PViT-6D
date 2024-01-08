import os
import csv
from dataclasses import dataclass, field
from pathlib import Path



@dataclass(slots=True)
class ResultWriter:
    output_dir:Path 
    results = field(default_factory=dict)

    def add_result(self, name, result):
        self.results[name] = result

    def write(self):
        header = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            # Write each row
            for result in self.results:
                # Flatten the rotation matrix and translation vector
                # And convert them to strings
                R_str = ' '.join(map(str, result['R'].flatten()))
                t_str = ' '.join(map(str, result['t'].flatten()))
                
                # Write the row to the csv file
                writer.writerow([
                    result['scene_id'],
                    result['im_id'],
                    result['obj_id'],
                    result['score'],
                    R_str,
                    t_str,
                    result['time']
                ])

    def add_result(self, pred, batch, dt):
        """
        {scene_id: {img_id: {obj_id: {score: score, R: R, t: t, time: time}}}}
        """
        scene_ids = batch["scene_id"].cpu().numpy()
        img_ids = batch["img_id"].cpu().numpy()
        obj_ids = batch["roi_cls"].cpu().numpy()
        rots = pred["rot"].cpu().numpy()
        trans = pred["trans"].cpu().numpy()
        time = dt[0] + dt[1] + dt[2] / batch["roi_cls"].shape[0]
        for idx, scene_id, img_id in enumerate(zip(scene_ids, img_ids)):
            self.results[scene_id][img_id] = {
                "obj_id": obj_ids[idx],
                "score": 1,
                "R": rots[idx],
                "t": trans[idx],
                "time": time,
            }



