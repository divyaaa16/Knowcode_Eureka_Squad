import { useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head'
import Link from 'next/link'

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to the static HTML page
    window.location.href = '/index.html';
  }, []);

  return null;
} 