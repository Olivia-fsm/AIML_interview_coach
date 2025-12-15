import { User, UserProfile, VisualHistoryItem, Submission, PrepPlan } from "../types";

const DB_KEY_USERS = 'ldc_users';
const DB_KEY_PROFILES = 'ldc_profiles';
const SESSION_KEY = 'ldc_session';

// --- Database Simulation ---

const getDb = () => {
  const users = JSON.parse(localStorage.getItem(DB_KEY_USERS) || '[]');
  const profiles = JSON.parse(localStorage.getItem(DB_KEY_PROFILES) || '{}');
  return { users, profiles };
};

const saveDb = (users: User[], profiles: Record<string, UserProfile>) => {
  localStorage.setItem(DB_KEY_USERS, JSON.stringify(users));
  localStorage.setItem(DB_KEY_PROFILES, JSON.stringify(profiles));
};

// --- Auth Services ---

export const loginUser = async (email: string, password: string): Promise<{user: User, profile: UserProfile}> => {
  // Simulate network delay
  await new Promise(r => setTimeout(r, 800));
  
  const { users, profiles } = getDb();
  const user = users.find((u: User) => u.email === email && u.password === password);
  
  if (!user) {
    throw new Error("Invalid email or password");
  }

  const profile = profiles[user.id] || initProfile(user.id);
  localStorage.setItem(SESSION_KEY, user.id);
  
  return { user, profile };
};

export const registerUser = async (email: string, password: string, name: string): Promise<{user: User, profile: UserProfile}> => {
  await new Promise(r => setTimeout(r, 800));
  
  const { users, profiles } = getDb();
  
  if (users.find((u: User) => u.email === email)) {
    throw new Error("User already exists");
  }

  const newUser: User = {
    id: Date.now().toString(),
    email,
    password, // Note: In production, hash this!
    name,
    joinedAt: Date.now(),
    avatar: `https://api.dicebear.com/7.x/notionists/svg?seed=${name}`
  };

  const newProfile = initProfile(newUser.id);
  
  users.push(newUser);
  profiles[newUser.id] = newProfile;
  
  saveDb(users, profiles);
  localStorage.setItem(SESSION_KEY, newUser.id);
  
  return { user: newUser, profile: newProfile };
};

export const logoutUser = () => {
  localStorage.removeItem(SESSION_KEY);
};

export const restoreSession = (): {user: User, profile: UserProfile} | null => {
  const userId = localStorage.getItem(SESSION_KEY);
  if (!userId) return null;
  
  const { users, profiles } = getDb();
  const user = users.find((u: User) => u.id === userId);
  const profile = profiles[userId];
  
  if (!user || !profile) {
    logoutUser();
    return null;
  }
  
  return { user, profile };
};

// --- Profile Management ---

const initProfile = (userId: string): UserProfile => ({
  userId,
  level: 1,
  xp: 0,
  likedProblemIds: [],
  visualHistory: [],
  gameHighScores: {},
  submissions: []
});

export const updateUserProfile = (userId: string, updates: Partial<UserProfile>): UserProfile => {
  const { users, profiles } = getDb();
  const currentProfile = profiles[userId] || initProfile(userId);
  
  const updatedProfile = { ...currentProfile, ...updates };
  profiles[userId] = updatedProfile;
  
  saveDb(users, profiles);
  return updatedProfile;
};

// --- Helper Actions ---

export const toggleLikeProblem = (userId: string, problemId: string): UserProfile => {
  const { profiles, users } = getDb();
  const profile = profiles[userId];
  if (!profile) throw new Error("Profile not found");
  
  const likes = new Set(profile.likedProblemIds);
  if (likes.has(problemId)) {
    likes.delete(problemId);
  } else {
    likes.add(problemId);
  }
  
  profile.likedProblemIds = Array.from(likes);
  profiles[userId] = profile;
  saveDb(users, profiles);
  return profile;
};

export const saveVisualGeneration = (userId: string, item: VisualHistoryItem): UserProfile => {
  const { profiles, users } = getDb();
  const profile = profiles[userId];
  if (!profile) return initProfile(userId); // Fallback
  
  profile.visualHistory = [item, ...profile.visualHistory];
  
  // Award XP for creativity
  profile.xp += 10;
  if (profile.xp >= profile.level * 100) {
      profile.level += 1;
      profile.xp = profile.xp - (profile.level - 1) * 100;
  }

  profiles[userId] = profile;
  saveDb(users, profiles);
  return profile;
};

export const saveGameScore = (userId: string, game: string, score: number): UserProfile => {
  const { profiles, users } = getDb();
  const profile = profiles[userId];
  if (!profile) return initProfile(userId);

  const currentHigh = profile.gameHighScores[game] || 0;
  if (score > currentHigh) {
    profile.gameHighScores[game] = score;
    // XP for new high score
    profile.xp += 20; 
  }
  
  profiles[userId] = profile;
  saveDb(users, profiles);
  return profile;
};

export const saveSubmission = (userId: string, submission: Submission): UserProfile => {
  const { profiles, users } = getDb();
  const profile = profiles[userId];
  if (!profile) return initProfile(userId);

  profile.submissions.push(submission);
  
  // XP Logic
  if (submission.feedback.correctnessScore > 80) {
      profile.xp += 50; // Big XP for correct solution
  } else {
      profile.xp += 5; // Participation XP
  }

  // Level Up Logic
  if (profile.xp >= profile.level * 100) {
      profile.level += 1;
      profile.xp = profile.xp - ((profile.level - 1) * 100); 
  }

  profiles[userId] = profile;
  saveDb(users, profiles);
  return profile;
};

export const saveUserPlan = (userId: string, plan: PrepPlan): UserProfile => {
    const { profiles, users } = getDb();
    const profile = profiles[userId];
    if (!profile) return initProfile(userId);
    
    profile.currentPlan = plan;
    profiles[userId] = profile;
    saveDb(users, profiles);
    return profile;
}