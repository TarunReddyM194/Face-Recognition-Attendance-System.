import streamlit as st
import json
import bcrypt
import os, pickle, io
from datetime import datetime, date
import pandas as pd
import face_recognition
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Face Attendance Admin", layout="wide")

# ---------- FILE PATHS ----------
ENC_DIR = "encodings"
ENC_FILE = os.path.join(ENC_DIR, "encodings.pkl")
FACULTY_FILE = "faculty.json"
TIMETABLE_FILE = "timetable.json"
SUB_FILE = "substitutions.json"
ATT_FILE = "attendance.csv"
SESSION_FILE = "session.json"
IMAGES_DIR = "images"
ADMINS_FILE = "admins.json"


# ---------- HELPERS ----------
def load_json_or_empty(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_enc_dir():
    if not os.path.exists(ENC_DIR):
        os.makedirs(ENC_DIR)


def load_encodings():
    if not os.path.exists(ENC_FILE):
        return {"encodings": [], "names": []}
    with open(ENC_FILE, "rb") as f:
        return pickle.load(f)


def save_encodings(data):
    ensure_enc_dir()
    with open(ENC_FILE, "wb") as f:
        pickle.dump(data, f)


def encode_image_file(uploaded_file, label):
    img_bytes = uploaded_file.read()
    img = face_recognition.load_image_file(io.BytesIO(img_bytes))
    faces = face_recognition.face_locations(img)
    if len(faces) == 0:
        raise ValueError("No face found in uploaded image.")
    enc = face_recognition.face_encodings(img, faces)[0]
    data = load_encodings()
    data["encodings"].append(enc)
    data["names"].append(label)
    save_encodings(data)
    return True


def add_image_file_to_images(uploaded_file, target_filename):
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    with open(os.path.join(IMAGES_DIR, target_filename), "wb") as f:
        f.write(uploaded_file.read())


def read_attendance_df():
    if not os.path.exists(ATT_FILE):
        return pd.DataFrame(columns=["Name", "Time", "Date", "Subject", "Faculty"])
    try:
        return pd.read_csv(ATT_FILE)
    except Exception:
        return pd.DataFrame(columns=["Name", "Time", "Date", "Subject", "Faculty"])


def write_attendance_df(df):
    df.to_csv(ATT_FILE, index=False)


# ---------- ADMIN MANAGEMENT ----------
def load_admins():
    # If admins.json missing, create default admin: Faculty / JU
    if not os.path.exists(ADMINS_FILE):
        default_pwd = "JU"
        hashed = bcrypt.hashpw(default_pwd.encode(), bcrypt.gensalt()).decode()
        admins = {
            "Faculty": {
                "pwd_hash": hashed,
                "display": "Faculty"
            }
        }
        with open(ADMINS_FILE, "w", encoding="utf-8") as f:
            json.dump({"admins": admins}, f, indent=2)
        return admins

    try:
        with open(ADMINS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("admins", {})
    except Exception:
        return {}


def save_admins(admins):
    with open(ADMINS_FILE, "w", encoding="utf-8") as f:
        json.dump({"admins": admins}, f, indent=2)


admins = load_admins()

# ---------- LOGIN STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_admin" not in st.session_state:
    st.session_state.current_admin = None

# ---------- LOGIN UI ----------
if not st.session_state.logged_in:
    st.title("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in admins:
            stored_hash = admins[username]["pwd_hash"].encode()
            if bcrypt.checkpw(password.encode(), stored_hash):
                st.session_state.logged_in = True
                st.session_state.current_admin = username
                st.success("Login successful! Loading dashboard...")
                st.rerun()
            else:
                st.error("Incorrect password")
        else:
            st.error("Username not found")

    st.stop()

# ---------- AFTER LOGIN ----------
st.sidebar.success(f"Logged in as: {st.session_state.current_admin}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.current_admin = None
    st.rerun()

st.title("📋 Face Attendance — Admin Dashboard")
# ---- Current Session Status ----
st.markdown("## 📚 Current Session Status")

session = load_json_or_empty(SESSION_FILE)

if session and session.get("active"):
    faculty_id = session.get("faculty", "Unknown")
    subject = session.get("subject", "Unknown")
    start_time = session.get("start_time", "")

    faculty_map = load_json_or_empty(FACULTY_FILE)
    faculty_name = faculty_map.get(faculty_id, {}).get("display", faculty_id)

    st.success("Session is ACTIVE")
    st.write(f"**Faculty:** {faculty_name}")
    st.write(f"**Subject:** {subject}")
    if start_time:
        st.write(f"**Started at:** {start_time}")
else:
    st.warning("No active session right now.")


faculty_map = load_json_or_empty(FACULTY_FILE)
timetable = load_json_or_empty(TIMETABLE_FILE)

st.sidebar.header("Quick Actions")
st.sidebar.markdown("Use tabs to navigate the system.")


# ---------- TABS ----------
tabs = st.tabs([
    "Live Camera Attendance",
    "Attendance",
    "Register",
    "Session Control",
    "People Management",
    "Substitutions",
    "Encodings & Export",
    "Admin Management"
])

# 0) Live Camera Attendance (info only)
with tabs[0]:
    st.header("Live Camera Attendance")
    st.write("Run `recognize.py` from your conda environment to start live face recognition.")
    st.code("conda activate face_env\npython recognize.py", language="bash")


# 1) Attendance tab
# 1) Attendance tab
with tabs[1]:
    st.header("Attendance")

    df = read_attendance_df()

    if df.empty:
        st.info("Attendance file is empty or not found.")
    else:
        # --- TODAY SUMMARY ---
        st.subheader("📌 Today’s Summary")

        today_str = date.today().isoformat()
        df_today = df[df["Date"] == today_str]

        if df_today.empty:
            st.info(f"No attendance marked for today ({today_str}) yet.")
        else:
            total_today = len(df_today)
            unique_names = sorted(df_today["Name"].unique())

            st.write(f"**Date:** {today_str}")
            st.write(f"**Total entries today:** {total_today}")
            st.write(f"**Students present today:** {', '.join(unique_names)}")

            st.markdown("**Last 5 entries today:**")
            st.text(df_today.tail(5).to_string(index=False))

        st.markdown("---")

        # --- FULL TABLE (TEXT) ---
        st.subheader("Full attendance table")
        st.text(df.to_string(index=False))

        st.markdown("---")
        st.subheader("Filter attendance")

        c1, c2, c3 = st.columns([1, 1, 1])
        date_sel = c1.date_input("Filter by date", value=date.today())
        subj_sel = c2.text_input("Filter by subject (empty = all)")
        fac_sel = c3.text_input("Filter by faculty (empty = all)")

        df_filtered = df.copy()
        df_filtered = df_filtered[df_filtered["Date"] == date_sel.isoformat()]

        if subj_sel.strip():
            df_filtered = df_filtered[
                df_filtered["Subject"].str.contains(subj_sel.strip(), case=False, na=False)
            ]
        if fac_sel.strip():
            df_filtered = df_filtered[
                df_filtered["Faculty"].str.contains(fac_sel.strip(), case=False, na=False)
            ]

        st.write(f"Showing {len(df_filtered)} rows for {date_sel.isoformat()}")

        if df_filtered.empty:
            st.info("No records found for the selected filters.")
        else:
            st.text(df_filtered.to_string(index=False))

            if st.button("Export filtered to CSV"):
                fn = f"attendance_{date_sel.isoformat()}.csv"
                df_filtered.to_csv(fn, index=False)
                st.success(f"Exported {fn} in current folder.")



# 2) Register tab (students & faculty)
with tabs[2]:
    st.header("Register new person (student / faculty)")

    role = st.radio("Role", ("Student", "Faculty"))
    label = st.text_input(
        "ID label (exact key used in encodings and faculty.json)",
        help="Example: Tarun_01 or Mayank_S"
    )
    uploaded = st.file_uploader(
        "Upload a clear frontal face image (jpg/png)", type=["jpg", "jpeg", "png"]
    )

    if st.button("Register & Encode"):
        if not label:
            st.error("Provide a label (ID) for this person.")
        elif not uploaded:
            st.error("Upload a face image first.")
        else:
            fname_ext = label if label.lower().endswith((".jpg", ".png", "jpeg")) else f"{label}.jpg"
            try:
                add_image_file_to_images(uploaded, fname_ext)
                with open(os.path.join(IMAGES_DIR, fname_ext), "rb") as f:
                    b = io.BytesIO(f.read())
                    b.seek(0)
                    encode_image_file(b, label)
                st.success(f"Encoded and registered {label}")

                if role == "Faculty":
                    fac = load_json_or_empty(FACULTY_FILE)
                    fac[label] = fac.get(label, {"display": label, "subject": "TBD"})
                    save_json(FACULTY_FILE, fac)
                    st.info("Added to faculty.json (edit display/subject in Session Control tab).")
            except Exception as e:
                st.error(f"Failed to register: {e}")

    st.markdown("---")
    st.subheader("Existing encodings")
    enc = load_encodings()
    st.write("Total encodings:", len(enc.get("names", [])))
    if st.button("Show encoded names"):
        st.write(enc.get("names", []))


# 3) Session Control
with tabs[3]:
    st.header("Session control (faculty-first workflow)")

    fac = load_json_or_empty(FACULTY_FILE)
    if not fac:
        st.warning("No faculty.json found or it is empty. Add faculty via Register tab.")
    else:
        st.write("Faculty list (click to start a session):")
        cols = st.columns(3)
        i = 0
        for fid, info in fac.items():
            display = info.get("display", fid)
            subj = info.get("subject", "TBD")
            if cols[i % 3].button(f"{display}\n({subj})", key=f"fac_{fid}"):
                now = datetime.now()
                session = {
                    "active": True,
                    "faculty": fid,
                    "subject": subj,
                    "start_time": now.isoformat(),
                    "expires_at": (now + pd.Timedelta(minutes=90)).isoformat()
                }
                save_json(SESSION_FILE, session)
                st.success(f"Session started by {display} for subject {subj}")
            i += 1

    st.markdown("Stop session:")
    if st.button("End session now"):
        if os.path.exists(SESSION_FILE):
            try:
                os.remove(SESSION_FILE)
            except Exception:
                save_json(SESSION_FILE, {"active": False})
        st.success("Session ended.")

    st.markdown("Edit faculty display/subject:")
    if fac:
        fid = st.selectbox("Choose faculty ID", options=list(fac.keys()))
        if fid:
            info = fac.get(fid, {})
            new_disp = st.text_input("Display name", value=info.get("display", fid))
            new_sub = st.text_input("Subject", value=info.get("subject", "TBD"))
            if st.button("Update faculty"):
                fac[fid]["display"] = new_disp
                fac[fid]["subject"] = new_sub
                save_json(FACULTY_FILE, fac)
                st.success("Updated faculty.json")


# 4) Substitutions tab
with tabs[4]:
    st.header("👥 People Management")

    # ---------- Student Management ----------
    st.subheader("🧑‍🎓 Student Management")

    enc_data = load_encodings()
    student_names = enc_data.get("names", [])

    if not student_names:
        st.info("No students found.")
    else:
        st.markdown("### Registered Students")
        for s in student_names:
            st.write("•", s)

        remove_student = st.selectbox(
            "Select student to remove",
            options=[""] + student_names
        )

        if st.button("❌ Remove Student"):
            if remove_student:
                idxs = [i for i, n in enumerate(enc_data["names"]) if n != remove_student]
                enc_data["names"] = [enc_data["names"][i] for i in idxs]
                enc_data["encodings"] = [enc_data["encodings"][i] for i in idxs]
                save_encodings(enc_data)
                st.success(f"Student '{remove_student}' removed.")
                st.rerun()
            else:
                st.warning("Please select a student.")

    st.markdown("---")

    # ---------- Faculty Management ----------
    st.subheader("🧑‍🏫 Faculty Management")

    faculty_data = load_json_or_empty(FACULTY_FILE)

    if not faculty_data:
        st.info("No faculty found.")
    else:
        st.markdown("### Registered Faculty")
        for fid, info in faculty_data.items():
            st.write(f"• {info.get('display', fid)} ({info.get('subject', 'TBD')})")

        remove_faculty = st.selectbox(
            "Select faculty to remove",
            options=[""] + list(faculty_data.keys())
        )

        if st.button("❌ Remove Faculty"):
            if remove_faculty:
                faculty_data.pop(remove_faculty)
                save_json(FACULTY_FILE, faculty_data)
                st.success(f"Faculty '{remove_faculty}' removed.")
                st.rerun()
            else:
                st.warning("Please select a faculty.")


# 5) Encodings & Export
with tabs[5]:
    st.header("Encodings & Export")
    enc = load_encodings()
    st.write("Encodings stored:", len(enc.get("names", [])))
    if st.button("Show names"):
        st.write(enc.get("names", []))

    st.markdown("---")
    st.subheader("Rebuild encodings from images folder")
    st.write("This will read all images in `images/` folder and create a fresh encodings.pkl.")

    if st.button("Rebuild encodings now"):
        names = []
        encs = []
        if not os.path.exists(IMAGES_DIR):
            st.error("images folder not found.")
        else:
            for root, dirs, files in os.walk(IMAGES_DIR):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        path = os.path.join(root, f)
                        try:
                            img = face_recognition.load_image_file(path)
                            locs = face_recognition.face_locations(img)
                            if len(locs) == 0:
                                st.warning(f"No face found in {path}; skipped.")
                                continue
                            e = face_recognition.face_encodings(img, locs)[0]
                            label = os.path.splitext(f)[0]
                            names.append(label)
                            encs.append(e)
                        except Exception as e:
                            st.warning(f"Error encoding {path}: {e}")
            data = {"encodings": encs, "names": names}
            save_encodings(data)
            st.success("Rebuilt encodings.pkl")

    st.markdown("---")
    st.subheader("Download attendance CSV")
    if os.path.exists(ATT_FILE):
        with open(ATT_FILE, "rb") as f:
            st.download_button("Download attendance.csv", f, file_name="attendance.csv")


# 6) Admin Management Tab
with tabs[6]:
    st.header("👥 Admin Account Management")

    # ---- Existing admins (NO pyarrow) ----
    st.subheader("Existing Admin Accounts")
    if admins:
        # show as plain text, one per line
        st.text("\n".join(admins.keys()))
    else:
        st.info("No admin accounts found.")

    st.markdown("---")
    st.subheader("➕ Add New Admin")

    new_user = st.text_input("New Admin Username")
    new_pass = st.text_input("New Admin Password", type="password")

    if st.button("Create Admin Account"):
        if not new_user or not new_pass:
            st.error("Username and password cannot be empty.")
        elif new_user in admins:
            st.error("Admin with this username already exists.")
        else:
            hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
            admins[new_user] = {
                "pwd_hash": hashed,
                "display": new_user
            }
            save_admins(admins)
            st.success(f"Admin '{new_user}' created successfully!")

    st.markdown("---")
    st.subheader("🗑 Remove Admin")

    if len(admins) > 1:
        remove_user = st.selectbox("Select admin to delete", list(admins.keys()))
        if st.button("Delete Admin"):
            if remove_user == st.session_state.current_admin:
                st.error("You cannot delete the admin currently logged in.")
            else:
                admins.pop(remove_user)
                save_admins(admins)
                st.success(f"Admin '{remove_user}' deleted successfully!")
    else:
        st.info("At least one admin must exist. Cannot delete the only admin.")

