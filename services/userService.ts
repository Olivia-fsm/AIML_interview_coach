import { supabase } from "./supabaseClient";
import { User, UserProfile, VisualHistoryItem, Submission, PrepPlan, CodeFeedback } from "../types";

// --- Helpers to map Supabase Data to App Types ---

const mapUser = (sbUser: any): User => ({
  id: sbUser.id,
  email: sbUser.email || '',
  name: sbUser.user_metadata?.full_name || 'Researcher',
  avatar: `https://api.dicebear.com/7.x/notionists/svg?seed=${sbUser.user_metadata?.full_name || sbUser.email}`,
  joinedAt: new Date(sbUser.created_at).getTime()
});

// Helper to sanitize feedback object to ensure all props are correct types (no objects where strings expected)
const sanitizeFeedback = (fb: any): CodeFeedback => {
    if (!fb || typeof fb !== 'object') {
        return { correctnessScore: 0, isCorrect: false, analysis: "N/A", improvements: [], timeComplexity: "?", spaceComplexity: "?" };
    }
    return {
        correctnessScore: typeof fb.correctnessScore === 'number' ? fb.correctnessScore : 0,
        isCorrect: !!fb.isCorrect,
        analysis: typeof fb.analysis === 'string' ? fb.analysis : "No analysis available.",
        improvements: Array.isArray(fb.improvements) ? fb.improvements.map((i:any) => String(i)) : [],
        timeComplexity: typeof fb.timeComplexity === 'string' ? fb.timeComplexity : "?",
        spaceComplexity: typeof fb.spaceComplexity === 'string' ? fb.spaceComplexity : "?"
    };
};

const mapProfile = (sbProfile: any, sbSubmissions: any[], sbVisuals: any[]): UserProfile => ({
  userId: sbProfile.id,
  level: sbProfile.level || 1,
  xp: sbProfile.xp || 0,
  likedProblemIds: sbProfile.liked_problem_ids || [],
  gameHighScores: sbProfile.game_high_scores || {},
  currentPlan: sbProfile.current_plan || undefined,
  // Map Submissions: SQL 'created_at' -> TS 'timestamp'
  submissions: sbSubmissions.map((s: any) => ({
    problemId: s.problem_id,
    code: s.code,
    feedback: sanitizeFeedback(s.feedback),
    timestamp: new Date(s.created_at).getTime()
  })),
  // Map Visuals
  visualHistory: sbVisuals.map((v: any) => ({
    id: v.id,
    type: v.type,
    mode: v.mode,
    prompt: v.prompt,
    // Fix: Using camelCase mediaUrl to match VisualHistoryItem interface
    mediaUrl: v.media_url,
    explanation: v.explanation,
    timestamp: new Date(v.created_at).getTime()
  }))
});

// --- Auth Services ---

export const registerUser = async (email: string, password: string, name: string): Promise<{user: User, profile: UserProfile}> => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: { full_name: name },
      emailRedirectTo: window.location.origin
    }
  });

  if (error) throw new Error(error.message);
  if (!data.user) throw new Error("Registration failed");

  const user = mapUser(data.user);
  const profile: UserProfile = {
      userId: user.id,
      level: 1, 
      xp: 0, 
      likedProblemIds: [], 
      visualHistory: [], 
      gameHighScores: {}, 
      submissions: [] 
  };

  return { user, profile };
};

export const loginUser = async (email: string, password: string): Promise<{user: User, profile: UserProfile}> => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  });

  if (error) throw new Error(error.message);
  if (!data.user) throw new Error("Login failed");

  const profile = await fetchFullProfile(data.user.id);
  return { user: mapUser(data.user), profile };
};

export const logoutUser = async () => {
  await supabase.auth.signOut();
};

export const restoreSession = async (): Promise<{user: User, profile: UserProfile} | null> => {
  const { data: { session } } = await supabase.auth.getSession();
  if (!session?.user) return null;

  try {
      const profile = await fetchFullProfile(session.user.id);
      return { user: mapUser(session.user), profile };
  } catch (e) {
      console.error("Failed to restore profile", e);
      return null;
  }
};

// --- Data Fetching ---

const fetchFullProfile = async (userId: string): Promise<UserProfile> => {
  // Parallel fetch for performance
  const [profileRes, subRes, visRes] = await Promise.all([
      supabase.from('profiles').select('*').eq('id', userId).maybeSingle(),
      supabase.from('submissions').select('*').eq('user_id', userId),
      supabase.from('visual_history').select('*').eq('user_id', userId).order('created_at', { ascending: false })
  ]);

  if (profileRes.error) console.error("Profile Fetch Error", profileRes.error);
  
  // If profile is missing (race condition), stub it
  const profileData = profileRes.data || { id: userId, level: 1, xp: 0, liked_problem_ids: [], game_high_scores: {} };

  return mapProfile(
      profileData, 
      subRes.data || [], 
      visRes.data || []
  );
};

// --- Actions (Write to DB) ---

export const toggleLikeProblem = async (userId: string, problemId: string): Promise<UserProfile> => {
  // 1. Get current state from DB or fresh fetch to be safe
  let currentProfile = await fetchFullProfile(userId);
  let likes = new Set<string>(currentProfile.likedProblemIds);

  if (likes.has(problemId)) likes.delete(problemId);
  else likes.add(problemId);

  const updatedLikes = Array.from(likes);

  // 2. Update DB using UPDATE (not upsert) to avoid RLS INSERT violations
  const { error } = await supabase.from('profiles').update({ liked_problem_ids: updatedLikes }).eq('id', userId);
  
  if (error) {
      console.error("Failed to toggle like (DB):", error);
      // Optimistic Return: Return the modified state so the UI updates regardless of DB error
      return { ...currentProfile, likedProblemIds: updatedLikes };
  }

  // 3. Return updated full profile from DB
  return fetchFullProfile(userId);
};

export const saveVisualGeneration = async (userId: string, item: VisualHistoryItem): Promise<UserProfile> => {
  // 1. Insert into table
  const { error: insertError } = await supabase.from('visual_history').insert({
      user_id: userId,
      type: item.type,
      mode: item.mode,
      prompt: item.prompt,
      media_url: item.mediaUrl,
      explanation: item.explanation
  });

  if (insertError) {
      console.error("Failed to save visual history (DB):", insertError);
  }

  // 2. Award XP
  const { data: profile } = await supabase.from('profiles').select('xp, level').eq('id', userId).maybeSingle();
  let xp = 0, level = 1;
  
  if (profile) {
      xp = (profile.xp || 0) + 10;
      level = profile.level || 1;
      if (xp >= level * 100) {
          level++;
          xp = xp - ((level - 1) * 100);
      }
      // Use UPDATE instead of UPSERT
      await supabase.from('profiles').update({ xp, level }).eq('id', userId);
  }

  // 3. Return Optimistic Result
  const freshProfile = await fetchFullProfile(userId);
  
  // FIXED: Deduplication check
  // Compare prompt, mode, and approximate timestamp (within 10 seconds)
  // This prevents displaying both the DB version (server time) and local version (client time)
  const exists = freshProfile.visualHistory.some(h => 
      h.prompt === item.prompt && 
      h.mode === item.mode &&
      Math.abs(h.timestamp - item.timestamp) < 10000 
  );
  
  if (!exists) {
      return {
          ...freshProfile,
          xp: xp || freshProfile.xp,
          level: level || freshProfile.level,
          visualHistory: [item, ...freshProfile.visualHistory]
      };
  }

  return freshProfile;
};

export const saveGameScore = async (userId: string, game: string, score: number): Promise<UserProfile> => {
  const { data: profile } = await supabase.from('profiles').select('game_high_scores, xp, level').eq('id', userId).maybeSingle();
  
  const scores = profile?.game_high_scores || {};
  const currentHigh = scores[game] || 0;
  
  if (score > currentHigh) {
      scores[game] = score;
      
      let xp = (profile?.xp || 0) + 20;
      let level = profile?.level || 1;
      
      if (xp >= level * 100) { level++; xp = xp - (level - 1) * 100; }

      // Use UPDATE
      const { error } = await supabase.from('profiles').update({ 
          game_high_scores: scores,
          xp, 
          level 
      }).eq('id', userId);
      
      if (error) console.error("Failed to save score:", error);
  }
  
  return fetchFullProfile(userId);
};

export const saveSubmission = async (userId: string, submission: Submission): Promise<UserProfile> => {
  // 1. Save submission
  const { error: insertError } = await supabase.from('submissions').insert({
      user_id: userId,
      problem_id: submission.problemId,
      code: submission.code,
      feedback: submission.feedback
  });

  if (insertError) console.error("Failed to save submission:", insertError);

  // 2. Calculate XP
  const { data: profile } = await supabase.from('profiles').select('xp, level').eq('id', userId).maybeSingle();
  
  let xp = (profile?.xp || 0);
  let level = (profile?.level || 1);
  
  if (submission.feedback.correctnessScore > 80) xp += 50;
  else xp += 5;

  if (xp >= level * 100) { level++; xp = xp - (level - 1) * 100; }
  
  // Use UPDATE
  await supabase.from('profiles').update({ xp, level }).eq('id', userId);

  // Optimistic Return with Deduplication
  const freshProfile = await fetchFullProfile(userId);
  
  // FIXED: Deduplication check using approximate timestamp and problem ID
  const exists = freshProfile.submissions.some(s => 
      s.problemId === submission.problemId &&
      Math.abs(s.timestamp - submission.timestamp) < 10000
  );
  
  if (!exists) {
      return {
          ...freshProfile,
          xp,
          level,
          submissions: [submission, ...freshProfile.submissions]
      };
  }

  return freshProfile;
};

export const saveUserPlan = async (userId: string, plan: PrepPlan): Promise<UserProfile> => {
  // Use UPDATE
  const { error } = await supabase.from('profiles').update({ current_plan: plan }).eq('id', userId);
  
  if (error) {
      console.error("Failed to save plan:", error);
      const current = await fetchFullProfile(userId);
      return { ...current, currentPlan: plan }; // Optimistic return
  }
  
  return fetchFullProfile(userId);
};

export const clearUserPlan = async (userId: string): Promise<UserProfile> => {
  const { error } = await supabase.from('profiles').update({ current_plan: null }).eq('id', userId);
  if (error) {
    console.error("Failed to clear plan:", error);
    throw error;
  }
  return fetchFullProfile(userId);
};
