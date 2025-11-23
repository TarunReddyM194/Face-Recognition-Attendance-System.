# recognize.py (faculty-first, session-based attendance) - quiet logging version
import io
import contextlib
import json
import os
import traceback
from datetime import datetime, time as dtime, timedelta

import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd

# Files
ENC_FILE = "encodings/encodings.pkl"
ATT_FILE = "attendance.csv"
TIMETABLE_FILE = "timetable.json"
SUB_FILE = "substitutions.json"
FACULTY_FILE = "faculty.json"
SESSION_FILE = "session.json"

# Config
SESSION_TIMEOUT_MIN = 90   # session expiry minutes after faculty starts

# ---------- helpers: load small json files ----------
def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Error loading", path, e)
        return None

# ---------- timetable/substitution ----------
def parse_hm(s):
    h, m = s.split(":")
    return dtime(int(h), int(m))

def get_subject_for_now(timetable):
    now = datetime.now()
    weekday = now.strftime("%A")
    current_hm = now.time()
    # check week mapping first
    week = timetable.get("week", {}) if timetable else {}
    day_map = week.get(weekday, {})
    for key, subj in day_map.items():
        try:
            start_s, end_s = key.split("-")
            start = parse_hm(start_s); end = parse_hm(end_s)
            if start <= current_hm <= end:
                return subj, key
        except Exception:
            continue
    # fallback to generic periods
    periods = timetable.get("periods", []) if timetable else []
    for p in periods:
        try:
            start = parse_hm(p["start"]); end = parse_hm(p["end"])
            if start <= current_hm <= end:
                return p.get("subject",""), f'{p["start"]}-{p["end"]}'
        except Exception:
            continue
    return "No Class", None

def load_substitutions():
    subs = load_json(SUB_FILE)
    return subs if subs else {}

def get_substitution_for_now():
    subs = load_substitutions()
    today = datetime.now().date().isoformat()
    if today not in subs:
        return None, None
    day_subs = subs[today]
    now_hm = datetime.now().time().strftime("%H:%M")
    for period_key, info in day_subs.items():
        try:
            start_s, end_s = period_key.split("-")
            if start_s <= now_hm <= end_s:
                return info.get("subject"), info.get("faculty")
        except Exception:
            continue
    return None, None

# ---------- faculty/session management ----------
def load_faculty():
    f = load_json(FACULTY_FILE)
    return f if f else {}

def load_session():
    s = load_json(SESSION_FILE)
    return s if s else {"active": False}

def save_session(s):
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)

def start_session(faculty_id, subject):
    """Start a session — if same faculty already active, only refresh expiry silently."""
    now = datetime.now()
    s = load_session()
    # if same faculty active, refresh expiry quietly
    if s.get("active") and s.get("faculty") == faculty_id:
        s["expires_at"] = (now + timedelta(minutes=SESSION_TIMEOUT_MIN)).isoformat()
        save_session(s)
        return False  # no new session started (refresh only)
    # else start new session
    s = {
        "active": True,
        "faculty": faculty_id,
        "subject": subject,
        "start_time": now.isoformat(),
        "expires_at": (now + timedelta(minutes=SESSION_TIMEOUT_MIN)).isoformat()
    }
    save_session(s)
    return True  # new session started

def expire_check():
    s = load_session()
    if s.get("active") and s.get("expires_at"):
        try:
            if datetime.fromisoformat(s["expires_at"]) < datetime.now():
                s["active"] = False
                save_session(s)
                print("Session expired automatically.")
        except Exception:
            pass

# ---------- attendance file ensure ----------
def ensure_attendance_file():
    if not os.path.exists(ATT_FILE):
        df = pd.DataFrame(columns=["Name","Time","Date","Subject","Faculty"])
        df.to_csv(ATT_FILE, index=False)
    else:
        try:
            df = pd.read_csv(ATT_FILE)
            changed = False
            for col in ["Name","Time","Date","Subject","Faculty"]:
                if col not in df.columns:
                    df[col] = ""
                    changed = True
            if changed:
                df.to_csv(ATT_FILE, index=False)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name","Time","Date","Subject","Faculty"])
            df.to_csv(ATT_FILE, index=False)

# ---------- mark attendance (uses session/substitution/timetable); returns True if new row added ----------
timetable = load_json(TIMETABLE_FILE)
faculty_map = load_faculty()

def mark_attendance(name, forced_subject=None, forced_faculty=None):
    now = datetime.now()
    date = now.date().isoformat()
    time_str = now.time().strftime("%H:%M:%S")

    # determine subject & faculty
    if forced_subject or forced_faculty:
        subject = forced_subject if forced_subject else None
        faculty_id = forced_faculty
    else:
        # check substitution first
        sub_subject, sub_faculty = get_substitution_for_now()
        if sub_subject or sub_faculty:
            subject = sub_subject if sub_subject else get_subject_for_now(timetable)[0]
            faculty_id = sub_faculty
        else:
            # check active session
            s = load_session()
            if s.get("active"):
                subject = s.get("subject")
                faculty_id = s.get("faculty")
            else:
                subject = get_subject_for_now(timetable)[0]
                faculty_id = None

    # resolve faculty display name
    faculty_display = None
    if faculty_id:
        teacher = faculty_map.get(faculty_id)
        if teacher:
            faculty_display = teacher.get("display", faculty_id)
        else:
            faculty_display = faculty_id
    else:
        fac_map = (timetable.get("faculty") if timetable else {}) or {}
        faculty_display = fac_map.get(subject, "Unknown")

    # read or create df
    try:
        df = pd.read_csv(ATT_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name","Time","Date","Subject","Faculty"])

    # avoid duplicate for same day+subject+name
    mask = (df["Name"]==name) & (df["Date"]==date) & (df["Subject"]==subject)
    if not mask.any():
        df.loc[len(df)] = [name, time_str, date, subject, faculty_display]
        df.to_csv(ATT_FILE, index=False)
        print(f"Attendance marked for {name} at {time_str} on {date} (Subject: {subject}, Faculty: {faculty_display})")
        return True
    # already marked -> be silent (no per-frame spam)
    return False

# ---------- load encodings ----------
if not os.path.exists(ENC_FILE):
    print("Encodings file not found. Run encode_faces.py first.")
    exit()

with open(ENC_FILE, "rb") as f:
    data = pickle.load(f)

known_encs = np.array(data.get("encodings", []))
known_names = np.array(data.get("names", []))

if len(known_encs) == 0:
    print("No encodings found. Run encode_faces.py and ensure images include faculty and students.")
    exit()

# prepare attendance file and run-time checks
ensure_attendance_file()

# ---------- camera loop ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to access camera.")
    exit()

print("Camera started. Press 'q' to quit.")

while True:
    expire_check()  # auto expire if needed

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small = small[:,:,::-1]
    rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_small, model="hog")

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    except Exception:
        print("Exception during face_encodings():")
        traceback.print_exc()
        face_encodings = []

    for face_encoding, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encs, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encs, face_encoding)
        name = "Unknown"
        if len(face_distances) > 0:
            best = np.argmin(face_distances)
            if matches[best]:
                name = known_names[best]

        # decide faculty vs student
        if name in faculty_map:
            # recognized a faculty -> start/refresh session
            subj = faculty_map[name].get("subject", get_subject_for_now(timetable)[0])
            started = start_session(name, subj)
            if started:
                display = faculty_map[name].get("display", name)
                print(f"Session started by {display} for subject {subj}")
            # show display name on video
            display = faculty_map[name].get("display", name)
            label = f"{display} (Faculty)"
        else:
            # student attendance — ONLY mark if a faculty session is active
            session = load_session()
            if session.get("active"):
                _ = mark_attendance(name)
            else:
                # session not active — do not mark unknown students or timetable fallback
                # optional: print a short message once (not every frame)
                # we'll silently ignore student until faculty starts
                pass
            label = name


        # draw box
        top, right, bottom, left = face_loc
        top*=4; right*=4; bottom*=4; left*=4
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
        cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
