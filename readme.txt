
              "subject_id": f"Sub-{sub_id:03d}",
                "age": ages[sub_id],
                "gender": genders[sub_id],
                "kinematic_input": kinematic.astype(np.float32), # [200, 18]
                "physiological_input": emg.astype(np.float32),   # [200, 4]
                "grf_label": grf.astype(np.float32),            # [200]
                "fatigue_label": fatigue_label,                  # 
                "technique_acc": 1 or 0   # 
