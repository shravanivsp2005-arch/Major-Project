import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { ArrowLeft, Loader2, ShieldCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Logo } from '@/components/common/Logo';
import { homePathForRole, useAuth } from '@/contexts/AuthContext';

export default function FirstAdminSetup() {
  const { bootstrapAdmin } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: '', email: '', password: '', phone: '' });
  const [loading, setLoading] = useState(false);
  const set = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const user = await bootstrapAdmin(form);
      toast.success('Initial administrator account created');
      navigate(homePathForRole(user.role), { replace: true });
    } catch (error) {
      toast.error(error?.response?.data?.detail || 'Could not set up the initial administrator account');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="flex flex-col justify-center px-5 py-10 sm:px-10">
        <div className="mx-auto w-full max-w-sm">
          <Link to="/login" className="mb-8 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground">
            <ArrowLeft className="h-4 w-4" /> Back to login
          </Link>
          <div className="lg:hidden"><Logo size="md" /></div>
          <h1 className="mt-6 font-display text-2xl font-bold text-foreground">Initial administrator setup</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Create the first administrator account to manage department officers and future administrators.
          </p>

          <form onSubmit={submit} className="mt-7 space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="name">Full name</Label>
              <Input id="name" required maxLength={120} value={form.name} onChange={set('name')} placeholder="Your name" />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" required value={form.email} onChange={set('email')} placeholder="you@example.com" />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" required minLength={6} value={form.password} onChange={set('password')} placeholder="At least 6 characters" />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="phone">Phone (optional)</Label>
              <Input id="phone" value={form.phone} onChange={set('phone')} placeholder="Phone number" />
            </div>

            <p className="rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground">
              This setup is available only once. Later administrator and department accounts must be created by an authenticated admin.
            </p>
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ShieldCheck className="mr-2 h-4 w-4" />}
              Create initial admin account
            </Button>
          </form>
        </div>
      </div>

      <div className="relative hidden overflow-hidden bg-sidebar lg:block">
        <div className="absolute inset-0 bg-grid opacity-[0.07]" />
        <div className="relative flex h-full flex-col justify-between p-12">
          <div className="flex justify-end"><Logo tone="light" size="lg" /></div>
          <div>
            <h2 className="font-display text-4xl font-bold leading-tight text-white">Secure city<br />administration.</h2>
            <p className="mt-4 max-w-md text-white/75">The first administrator sets up the team. All future privileged accounts require admin approval.</p>
          </div>
          <p className="text-xs text-white/50">One-time administrator setup</p>
        </div>
      </div>
    </div>
  );
}
