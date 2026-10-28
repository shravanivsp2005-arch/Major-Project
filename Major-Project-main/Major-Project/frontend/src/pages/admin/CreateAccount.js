import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { Loader2, UserPlus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DEPARTMENTS } from '@/lib/constants';
import { dataApi } from '@/lib/api';

export default function AdminCreateAccount() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: '',
    phone: '',
    role: 'officer',
    department: DEPARTMENTS[0],
  });
  const [loading, setLoading] = useState(false);
  const set = (key) => (value) => setForm((current) => ({ ...current, [key]: value }));

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      await dataApi.createUser({
        ...form,
        department: form.role === 'officer' ? form.department : null,
      });
      toast.success(`${form.role === 'admin' ? 'Administrator' : 'Department officer'} account created`);
      navigate('/admin/users');
    } catch (error) {
      toast.error(error?.response?.data?.detail || 'Could not create account');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <div>
        <h1 className="font-display text-2xl font-bold text-foreground">Create an account</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Create an administrator or assign an officer to a department.
        </p>
      </div>

      <form onSubmit={submit} className="space-y-5 rounded-2xl border border-border bg-card p-5 card-shadow sm:p-7">
        <div className="space-y-1.5">
          <Label htmlFor="role">Account type</Label>
          <Select value={form.role} onValueChange={set('role')}>
            <SelectTrigger id="role"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="admin">Administrator</SelectItem>
              <SelectItem value="officer">Department officer</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {form.role === 'officer' && (
          <div className="space-y-1.5">
            <Label htmlFor="department">Department</Label>
            <Select value={form.department} onValueChange={set('department')}>
              <SelectTrigger id="department"><SelectValue placeholder="Choose a department" /></SelectTrigger>
              <SelectContent>
                {DEPARTMENTS.map((department) => (
                  <SelectItem key={department} value={department}>{department}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        <div className="space-y-1.5">
          <Label htmlFor="name">Full name</Label>
          <Input id="name" required maxLength={120} value={form.name} onChange={(event) => set('name')(event.target.value)} placeholder="Full name" />
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="email">Email</Label>
          <Input id="email" type="email" required value={form.email} onChange={(event) => set('email')(event.target.value)} placeholder="name@example.com" />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="password">Initial password</Label>
            <Input id="password" type="password" required minLength={6} value={form.password} onChange={(event) => set('password')(event.target.value)} placeholder="At least 6 characters" />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="phone">Phone (optional)</Label>
            <Input id="phone" value={form.phone} onChange={(event) => set('phone')(event.target.value)} placeholder="Phone number" />
          </div>
        </div>

        <p className="rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground">
          Share the initial password with the account holder securely. They can sign in at the regular login page.
        </p>

        <Button type="submit" disabled={loading}>
          {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <UserPlus className="mr-2 h-4 w-4" />}
          Create account
        </Button>
      </form>
    </div>
  );
}
