import NextAuth, { NextAuthOptions, Session, User } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import GitHubProvider from "next-auth/providers/github";
import { JWT } from "next-auth/jwt";

export default NextAuth({
	providers: [
		GoogleProvider({
			clientId: process.env.GOOGLE_CLIENT_ID!,
			clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
		}),
		GitHubProvider({
			clientId: process.env.GITHUB_CLIENT_ID!,
			clientSecret: process.env.GITHUB_CLIENT_SECRET!,
		}),
	],
	callbacks: {
		async session({ session, token, user }) {
			// Extend session.user type
			(session.user as any).id = user?.id || token?.sub || user?.email ||"";
			// Type guard for login property
			const userLogin =
				user && typeof user === "object" && "login" in user
					? (user as any).login
					: undefined;
			const tokenLogin =
				token && typeof token === "object" && "login" in token
					? (token as any).login
					: undefined;
			if (userLogin) {
				(session.user as any).github_username = userLogin;
			} else if (tokenLogin) {
				(session.user as any).github_username = tokenLogin;
			} else {
				(session.user as any).github_username = "";
			}
			return session;
		},
		async redirect({ url, baseUrl }) {
			if (url.startsWith("/")) return `${baseUrl}${url}`;
			else if (new URL(url).origin === baseUrl) return url;
			return baseUrl;
		},
	},
	pages: {
		signIn: "/auth/signin",
		error: "/auth/error",
		signOut: "/auth/signout",
	},
});
 