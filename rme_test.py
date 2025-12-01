# -----------------------
# IMPORTS
# -----------------------
from psychopy import visual, event, core, gui, data
import os
import random
import csv
from PIL import Image
from datetime import datetime

# -----------------------
# PARTICIPANT INFO (PsychoPy GUI)
# -----------------------
participant_info = {"Participant ID": "", "Condition": ["vanilla", "neutral"]}

dlg = gui.DlgFromDict(participant_info, title="Participant Info")
if not dlg.OK:
    core.quit()  # Exit if user cancels

participant_id = participant_info["Participant ID"]
condition = participant_info["Condition"].lower()

# -----------------------
# DATA FILE SETUP
# -----------------------
if not os.path.exists("data"):
    os.makedirs("data")

timestamp_str = data.getDateStr()
filename = f"data/{participant_id}_{timestamp_str}.csv"
logfile = f"data/{participant_id}_{timestamp_str}_keylog.csv"

# -----------------------
# WINDOW SETUP
# -----------------------
win = visual.Window(size=(1000, 800), color="grey", units="pix")

# -----------------------
# STIMULI SETUP
# -----------------------
stim_folder = "FACES"
all_files = [f for f in os.listdir(stim_folder) if f.endswith(".jpg")]

emotion_map = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutrality",
    "s": "sadness"
}

def crop_eyes(img_path):
    """Crop face image horizontally to show only the eye region and return PIL Image object."""
    img = Image.open(img_path)
    w, h = img.size
    top = int(h * 0.35)
    bottom = int(h * 0.55)
    left = int(w * 0.1)
    right = int(w * 0.9)
    return img.crop((left, top, right, bottom))



# Prepare stimuli list
stimuli = []
for f in all_files:
    if len(f) >= 9:
        code = f[8].lower()
        if code in emotion_map:
            stimuli.append((os.path.join(stim_folder, f), emotion_map[code]))
            
random.shuffle(stimuli)

# -----------------------
# TRIAL SETUP (take 20 images)
# -----------------------
num_trials = 20
if len(stimuli) > num_trials:
    stimuli = random.sample(stimuli, num_trials)

random.shuffle(stimuli)

# -----------------------
# TRIAL LOOP
# -----------------------
clock = core.Clock()

with open(filename, "w", newline="", encoding="utf-8") as csvfile, \
     open(logfile, "w", newline="", encoding="utf-8") as logcsv:

    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "ParticipantID", "Condition", "Trial", "Image",
        "CorrectEmotion", "Response", "Accuracy", "RT"
    ])

    log_writer = csv.writer(logcsv, quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(["Trial", "KeyPressed", "TimeStamp", "RT"])

    for trial_num, (stim_file, correct_emotion) in enumerate(stimuli, start=1):
        # Crop image on the fly
        cropped_img = crop_eyes(stim_file)
        temp_path = "temp_cropped.jpg"
        cropped_img.save(temp_path)
        stim = visual.ImageStim(win, image=temp_path, size=(600, 200))
        stim.draw()

        # Prepare 4-choice options
        options = [correct_emotion]
        while len(options) < 4:
            cand = random.choice(list(emotion_map.values()))
            if cand not in options:
                options.append(cand)
        random.shuffle(options)

        # Display options as text
        y_positions = [-200, -260, -320, -380]
        for i, opt in enumerate(options):
            visual.TextStim(win, text=f"{i+1}. {opt}", pos=(0, y_positions[i]), height=30, color="white").draw()

        win.flip()
        clock.reset()

        # Wait for response and log keypresses
        keys = event.waitKeys(keyList=["1", "2", "3", "4", "escape"], timeStamped=clock)
        for key, rt in keys:
            log_writer.writerow([trial_num, key, datetime.now().isoformat(), rt])

        if "escape" in [k for k, t in keys]:
            print("Experiment aborted by user.")
            break

        choice = int(keys[0][0]) - 1
        response = options[choice]
        accuracy = 1 if response == correct_emotion else 0

        writer.writerow([
            participant_id,
            condition,
            trial_num,
            stim_file,
            correct_emotion,
            response,
            accuracy,
            keys[0][1]
        ])

        core.wait(0.5)

# -----------------------
# CLEANUP
# -----------------------
win.close()
core.quit()
if os.path.exists("temp_cropped.jpg"):
    os.remove("temp_cropped.jpg")
